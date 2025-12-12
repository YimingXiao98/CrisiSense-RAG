#!/usr/bin/env python
"""
Compute mean flood depth per ZIP code from FEMA Harvey flood depth raster.
Outputs a JSON file with flood_depth_m per ZIP for use as ground truth.
"""
import json
from pathlib import Path
from osgeo import gdal, ogr
import numpy as np
from collections import defaultdict
from loguru import logger

gdal.UseExceptions()
ogr.UseExceptions()


def load_zip_boundaries(shapefile_path: str):
    """Load ZIP code polygon boundaries."""
    ds = ogr.Open(shapefile_path)
    if not ds:
        raise ValueError(f"Cannot open {shapefile_path}")
    
    layer = ds.GetLayer()
    boundaries = {}
    
    for feature in layer:
        zip_code = feature.GetField("ZCTA5CE10") or feature.GetField("GEOID10") or feature.GetField("ZIP")
        if zip_code:
            geom = feature.GetGeometryRef()
            boundaries[str(zip_code)] = geom.Clone()
    
    logger.info(f"Loaded {len(boundaries)} ZIP boundaries")
    return boundaries


def compute_zonal_stats(raster_path: str, zip_boundaries: dict, sample_size: int = 1000):
    """Compute mean flood depth per ZIP using sampling."""
    ds = gdal.Open(raster_path)
    if not ds:
        raise ValueError(f"Cannot open {raster_path}")
    
    band = ds.GetRasterBand(1)
    gt = ds.GetGeoTransform()  # [x_origin, pixel_width, 0, y_origin, 0, pixel_height]
    nodata = band.GetNoDataValue()
    
    logger.info(f"Raster size: {ds.RasterXSize} x {ds.RasterYSize}")
    logger.info(f"Geotransform: {gt}")
    logger.info(f"NoData value: {nodata}")
    
    results = {}
    
    for zip_code, geom in zip_boundaries.items():
        # Get bounding box
        env = geom.GetEnvelope()  # (minX, maxX, minY, maxY)
        
        # Convert to pixel coordinates
        px_min = int((env[0] - gt[0]) / gt[1])
        px_max = int((env[1] - gt[0]) / gt[1])
        py_min = int((env[3] - gt[3]) / gt[5])  # Note: gt[5] is negative
        py_max = int((env[2] - gt[3]) / gt[5])
        
        # Clip to raster bounds
        px_min = max(0, px_min)
        px_max = min(ds.RasterXSize, px_max)
        py_min = max(0, py_min)
        py_max = min(ds.RasterYSize, py_max)
        
        if px_max <= px_min or py_max <= py_min:
            results[zip_code] = {"mean_depth_m": 0.0, "max_depth_m": 0.0, "flooded_pct": 0.0}
            continue
        
        # Read window
        width = px_max - px_min
        height = py_max - py_min
        
        try:
            data = band.ReadAsArray(px_min, py_min, width, height)
            if data is None:
                results[zip_code] = {"mean_depth_m": 0.0, "max_depth_m": 0.0, "flooded_pct": 0.0}
                continue
            
            # Mask nodata
            if nodata is not None:
                valid = data != nodata
            else:
                valid = data > 0
            
            valid_data = data[valid]
            
            if len(valid_data) > 0:
                mean_depth = float(np.mean(valid_data))
                max_depth = float(np.max(valid_data))
                flooded_pct = float(np.sum(valid) / data.size * 100)
            else:
                mean_depth = 0.0
                max_depth = 0.0
                flooded_pct = 0.0
            
            results[zip_code] = {
                "mean_depth_m": round(mean_depth, 3),
                "max_depth_m": round(max_depth, 3),
                "flooded_pct": round(flooded_pct, 2)
            }
            
        except Exception as e:
            logger.warning(f"Error processing {zip_code}: {e}")
            results[zip_code] = {"mean_depth_m": 0.0, "max_depth_m": 0.0, "flooded_pct": 0.0}
    
    return results


def main():
    # Paths
    gdb_path = "data/raw/fema_flood_depths/Harvey_Depths_3m_Final.gdb"
    
    # Try to find ZIP boundaries
    zip_shapefile = "data/raw/zip_boundaries/tl_2017_us_zcta510.shp"
    
    if not Path(zip_shapefile).exists():
        logger.warning(f"ZIP shapefile not found: {zip_shapefile}")
        logger.info("Will compute stats for entire raster instead")
        
        # Just read the full raster stats
        ds = gdal.Open(gdb_path)
        band = ds.GetRasterBand(1)
        
        # Sample the raster
        logger.info("Sampling raster to compute overall statistics...")
        sample_x = np.linspace(0, ds.RasterXSize-1, 1000, dtype=int)
        sample_y = np.linspace(0, ds.RasterYSize-1, 1000, dtype=int)
        
        values = []
        for y in sample_y[:100]:
            row = band.ReadAsArray(0, int(y), ds.RasterXSize, 1)
            if row is not None:
                nodata = band.GetNoDataValue() or 0
                valid = row[row != nodata]
                if len(valid) > 0:
                    values.extend(valid.flatten()[:1000])
        
        values = np.array(values)
        logger.info(f"Sampled {len(values)} pixels")
        logger.info(f"Mean depth: {np.mean(values):.3f} m")
        logger.info(f"Max depth: {np.max(values):.3f} m")
        logger.info(f"Min depth: {np.min(values):.3f} m")
        
        return
    
    # Load ZIP boundaries
    zip_boundaries = load_zip_boundaries(zip_shapefile)
    
    # Compute zonal stats
    results = compute_zonal_stats(gdb_path, zip_boundaries)
    
    # Save results
    output_path = Path("data/processed/flood_depth_by_zip.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.success(f"Saved flood depth stats for {len(results)} ZIPs to {output_path}")
    
    # Print top 10 by mean depth
    sorted_zips = sorted(results.items(), key=lambda x: x[1]["mean_depth_m"], reverse=True)
    print("\nTop 10 ZIPs by mean flood depth:")
    for zip_code, stats in sorted_zips[:10]:
        print(f"  {zip_code}: {stats['mean_depth_m']:.3f}m (max: {stats['max_depth_m']:.3f}m, {stats['flooded_pct']:.1f}% flooded)")


if __name__ == "__main__":
    main()
