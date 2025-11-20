"""Spatial indexing utilities using hash map for O(1) access."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from ..dataio.loaders import load_parquet_table


@dataclass
class SpatialIndex:
    """In-memory spatial index for imagery and sensors."""

    imagery: List[dict]
    sensors: List[dict]
    _zip_index: Dict[str, List[dict]] = field(init=False, default_factory=dict)
    _sensor_zip_index: Dict[str, List[dict]] = field(init=False, default_factory=dict)

    def __post_init__(self):
        # Index imagery by ZIP
        for tile in self.imagery:
            z = tile.get("zip")
            if z:
                if z not in self._zip_index:
                    self._zip_index[z] = []
                self._zip_index[z].append(tile)
        
        # Index sensors by ZIP
        for sensor in self.sensors:
            z = sensor.get("zip")
            if z:
                if z not in self._sensor_zip_index:
                    self._sensor_zip_index[z] = []
                self._sensor_zip_index[z].append(sensor)

    @classmethod
    def from_parquet(cls, imagery_path: Path, sensors_path: Path) -> "SpatialIndex":
        return cls(load_parquet_table(imagery_path), load_parquet_table(sensors_path))

    def get_tiles_by_zip(self, zip_code: str, start: datetime, end: datetime, k: int) -> List[dict]:
        """Return imagery tiles filtered by ZIP/time with graceful fallback."""

        if k <= 0:
            return []

        tiles = self._filter_tiles(zip_code=zip_code, start=start, end=end)
        if tiles:
            return tiles[:k]

        if zip_code:
            logger.warning("No imagery tiles for requested ZIP/time window; falling back to time-only filter", zip=zip_code)
        time_only = self._filter_tiles(zip_code=None, start=start, end=end)
        if time_only:
            return time_only[:k]

        logger.warning("No imagery tiles within time window; returning most recent globally", zip=zip_code)
        latest = sorted(
            [
                (self._parse_timestamp(tile.get("timestamp")), tile)
                for tile in self.imagery
                if self._parse_timestamp(tile.get("timestamp"))
            ],
            key=lambda item: item[0],
            reverse=True,
        )
        return [tile for _, tile in latest[:k]]

    def nearest_sensors_by_zip(self, zip_code: str, n: int = 3) -> List[dict]:
        sensors = self._sensor_zip_index.get(zip_code, [])
        if not sensors and not zip_code:
             sensors = self.sensors
             
        sensors_sorted = sorted(
            sensors, 
            key=lambda s: s.get("timestamp", ""), 
            reverse=True
        )
        return sensors_sorted[:n]

    def _filter_tiles(self, zip_code: Optional[str], start: datetime, end: datetime) -> List[dict]:
        # Use index if ZIP is provided
        candidates = self._zip_index.get(zip_code, []) if zip_code else self.imagery
        
        matches = []
        for tile in candidates:
            ts = self._parse_timestamp(tile.get("timestamp"))
            if ts is None or ts < start or ts > end:
                continue
            matches.append((ts, tile))
        matches.sort(key=lambda item: item[0])
        return [tile for _, tile in matches]

    @staticmethod
    def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            try:
                return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return None
