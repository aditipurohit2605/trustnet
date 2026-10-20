"""
Image Bank indexer and visual reuse detector for TrustNet.

Integrates:
1. Perceptual Hashing (pHash via ImageHash): fast detection of exact duplicates,
   resizing, recompression, light cropping, and color jitter.
2. Semantic Embeddings (OpenCLIP + FAISS IndexFlatIP): deep visual feature search
   capable of identifying re-used imagery even under heavier alterations or framing.
3. Metadata tracking: associates each bank image with known origin, context, and source.

Decision Rule:
An image is identified as a reuse match if:
- (a) Perceptual hash Hamming distance <= phash_match_threshold (exact/near-duplicate), OR
- (b) CLIP cosine similarity >= embedding_match_threshold (semantic visual reuse).
Candidates are ranked by lowest pHash distance, then highest CLIP cosine similarity.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import imagehash
import numpy as np
from PIL import Image

from trustnet.schemas import ReuseMatch
from trustnet.vision.clip_scorer import ClipScorer, _to_pil_image

logger = logging.getLogger(__name__)


class ImageBank:
    """
    Reverse image lookup index combining perceptual hashing and FAISS dense visual embeddings.
    """

    def __init__(
        self,
        clip_scorer: Optional[ClipScorer] = None,
        phash_size: int = 8,
        phash_match_threshold: int = 10,
        embedding_match_threshold: float = 0.85,
    ):
        """
        Args:
            clip_scorer: ClipScorer instance used to extract image embeddings.
            phash_size: Size of the perceptual hash matrix (default 8 -> 64-bit hash).
            phash_match_threshold: Maximum Hamming distance to consider a pHash match (0 to 64).
            embedding_match_threshold: Minimum cosine similarity for dense visual match (0 to 1).
        """
        self.clip_scorer = clip_scorer
        self.phash_size = phash_size
        self.phash_match_threshold = phash_match_threshold
        self.embedding_match_threshold = embedding_match_threshold

        # In-memory index structures
        self.bank_ids: List[str] = []
        self.metadata_records: Dict[str, Dict[str, Any]] = {}
        self.phashes: Dict[str, imagehash.ImageHash] = {}

        # FAISS IndexFlatIP: inner product equals cosine similarity on L2-normalized vectors
        self.embedding_dim: int = 512
        self.faiss_index: Optional[faiss.IndexFlatIP] = None

    def add_image(
        self,
        bank_id: str,
        image: Union[Image.Image, np.ndarray, bytes, str, Path],
        known_context: Optional[str] = None,
        source: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> None:
        """
        Adds a single reference image to the bank index.
        """
        pil_img = _to_pil_image(image)

        # 1. Compute perceptual hash
        h = imagehash.phash(pil_img, hash_size=self.phash_size)
        self.phashes[bank_id] = h

        # 2. Store metadata
        self.metadata_records[bank_id] = {
            "bank_id": bank_id,
            "path": str(file_path or ""),
            "known_context": known_context or "",
            "source": source or "",
        }

        # 3. Extract and index CLIP embedding
        if self.clip_scorer is not None:
            emb = self.clip_scorer.encode_image(pil_img)  # [1, D]
            if self.faiss_index is None:
                self.embedding_dim = emb.shape[1]
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)

            # Ensure contiguous float32 array for FAISS
            emb_f32 = np.ascontiguousarray(emb, dtype=np.float32)
            self.faiss_index.add(emb_f32)

        self.bank_ids.append(bank_id)

    def build_from_directory(
        self,
        bank_dir: Union[str, Path],
        metadata_filename: str = "metadata.jsonl",
    ) -> "ImageBank":
        """
        Scans a directory containing images and a metadata JSONL file to populate the index.
        """
        bank_path = Path(bank_dir)
        if not bank_path.exists():
            logger.warning(f"Image bank directory does not exist: {bank_path}")
            return self

        meta_file = bank_path / metadata_filename
        metadata_by_id: Dict[str, Dict[str, Any]] = {}
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        metadata_by_id[record.get("bank_id", "")] = record

        # Supported image extensions
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        image_files = [p for p in bank_path.iterdir() if p.suffix.lower() in extensions]

        logger.info(f"Indexing {len(image_files)} bank images from {bank_path}...")
        for img_path in image_files:
            bank_id = img_path.stem
            meta = metadata_by_id.get(bank_id, {})
            try:
                self.add_image(
                    bank_id=bank_id,
                    image=img_path,
                    known_context=meta.get("known_context"),
                    source=meta.get("source"),
                    file_path=str(img_path),
                )
            except Exception as e:
                logger.warning(f"Failed to index image {img_path}: {e}")

        logger.info(f"ImageBank built successfully with {len(self.bank_ids)} indexed items.")
        return self

    def save(self, index_dir: Union[str, Path]) -> None:
        """
        Serializes the index (FAISS binary, hashes, and metadata) to disk.
        """
        out_dir = Path(index_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save FAISS index
        if self.faiss_index is not None:
            faiss_file = out_dir / "clip_faiss.index"
            faiss.write_index(self.faiss_index, str(faiss_file))

        # 2. Save perceptual hashes and bank_id mapping
        hashes_data = {
            "phash_size": self.phash_size,
            "bank_ids": self.bank_ids,
            "hashes": {bid: str(h) for bid, h in self.phashes.items()},
        }
        with open(out_dir / "phashes.json", "w", encoding="utf-8") as f:
            json.dump(hashes_data, f, indent=2)

        # 3. Save metadata JSONL
        with open(out_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
            for bid in self.bank_ids:
                f.write(json.dumps(self.metadata_records.get(bid, {"bank_id": bid})) + "\n")

        logger.info(f"Saved ImageBank index to {out_dir}")

    @classmethod
    def load(
        cls,
        index_dir: Union[str, Path],
        clip_scorer: Optional[ClipScorer] = None,
        phash_match_threshold: int = 10,
        embedding_match_threshold: float = 0.85,
    ) -> "ImageBank":
        """
        Loads an existing index from disk without recomputing hashes or embeddings.
        """
        in_dir = Path(index_dir)
        hashes_file = in_dir / "phashes.json"
        if not hashes_file.exists():
            raise FileNotFoundError(f"ImageBank hashes file not found: {hashes_file}")

        with open(hashes_file, "r", encoding="utf-8") as f:
            hdata = json.load(f)

        bank = cls(
            clip_scorer=clip_scorer,
            phash_size=hdata.get("phash_size", 8),
            phash_match_threshold=phash_match_threshold,
            embedding_match_threshold=embedding_match_threshold,
        )

        bank.bank_ids = hdata.get("bank_ids", [])
        bank.phashes = {
            bid: imagehash.hex_to_hash(hex_str)
            for bid, hex_str in hdata.get("hashes", {}).items()
        }

        # Load FAISS index if present
        faiss_file = in_dir / "clip_faiss.index"
        if faiss_file.exists():
            bank.faiss_index = faiss.read_index(str(faiss_file))

        # Load metadata JSONL
        meta_file = in_dir / "metadata.jsonl"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        bank.metadata_records[rec.get("bank_id", "")] = rec

        logger.info(f"Loaded ImageBank from {in_dir} with {len(bank.bank_ids)} images.")
        return bank

    def search(
        self,
        image: Union[Image.Image, np.ndarray, bytes, str, Path],
        top_k: int = 5,
    ) -> List[ReuseMatch]:
        """
        Searches the image bank for near-duplicate or reused imagery.

        Decision Rule Details:
        - Exact/Near duplicate match: pHash Hamming distance <= phash_match_threshold.
        - Semantic visual reuse match: CLIP cosine similarity >= embedding_match_threshold.
        - Candidate items meeting either threshold are included.
        - Candidates are sorted primarily by pHash distance (ascending), then by
          embedding similarity (descending).

        Args:
            image: Query image in RAM or path.
            top_k: Maximum number of matches to return.

        Returns:
            List of ReuseMatch schema objects.
        """
        if not self.bank_ids:
            return []

        pil_img = _to_pil_image(image)

        # 1. Compute query perceptual hash and pairwise Hamming distances
        q_phash = imagehash.phash(pil_img, hash_size=self.phash_size)
        phash_distances: Dict[str, int] = {}
        for bid, h in self.phashes.items():
            # The '-' operator computes the bitwise Hamming distance between two pHashes
            phash_distances[bid] = int(q_phash - h)

        # 2. Compute query CLIP embedding and retrieve similarities via FAISS
        embedding_sims: Dict[str, float] = {}
        if self.clip_scorer is not None and self.faiss_index is not None and len(self.bank_ids) > 0:
            q_emb = self.clip_scorer.encode_image(pil_img)  # [1, D]
            q_emb_f32 = np.ascontiguousarray(q_emb, dtype=np.float32)

            k_search = min(len(self.bank_ids), max(top_k * 2, 20))
            sims, idxs = self.faiss_index.search(q_emb_f32, k_search)

            for sim_val, idx_val in zip(sims[0], idxs[0]):
                if 0 <= idx_val < len(self.bank_ids):
                    bid = self.bank_ids[idx_val]
                    embedding_sims[bid] = float(np.clip(sim_val, -1.0, 1.0))

        # Default embedding similarity to 0.0 for any uncomputed items
        for bid in self.bank_ids:
            if bid not in embedding_sims:
                embedding_sims[bid] = 0.0

        # 3. Apply the Composite Decision Rule:
        # Include candidate if pHash distance <= threshold OR embedding similarity >= threshold
        matched_results: List[ReuseMatch] = []
        for bid in self.bank_ids:
            dist = phash_distances.get(bid, 999)
            sim = embedding_sims.get(bid, 0.0)

            is_hash_match = dist <= self.phash_match_threshold
            is_dense_match = sim >= self.embedding_match_threshold

            if is_hash_match or is_dense_match:
                meta = self.metadata_records.get(bid, {})
                matched_results.append(
                    ReuseMatch(
                        bank_id=bid,
                        phash_distance=dist,
                        embedding_similarity=round(sim, 4),
                        known_context=meta.get("known_context"),
                        source=meta.get("source"),
                    )
                )

        # Sort results: pHash distance ascending (0 is exact duplicate), then similarity descending
        matched_results.sort(
            key=lambda m: (m.phash_distance, -m.embedding_similarity)
        )

        return matched_results[:top_k]
