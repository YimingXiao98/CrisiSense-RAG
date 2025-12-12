#!/usr/bin/env python3
"""
Build ZIP-to-coordinate lookup for Harris County and surrounding area.

This creates a JSON file mapping ZIP codes to their centroids for:
1. Geo-boosting text retrieval scores
2. Geo-filtering imagery by distance from query ZIP

Usage:
    python scripts/build_zip_coordinates.py
"""

import json
from pathlib import Path

# Harris County and Greater Houston area ZIP codes with approximate centroids
# Source: USPS ZIP code data / Census ZCTA centroids
# These coordinates are approximate centroids for each ZIP

HOUSTON_ZIP_CENTROIDS = {
    # Houston Core (Harris County)
    "77002": {"lat": 29.7604, "lon": -95.3698, "name": "Downtown Houston"},
    "77003": {"lat": 29.7433, "lon": -95.3485, "name": "East Downtown"},
    "77004": {"lat": 29.7286, "lon": -95.3621, "name": "Third Ward"},
    "77005": {"lat": 29.7172, "lon": -95.4231, "name": "West University Place"},
    "77006": {"lat": 29.7438, "lon": -95.3905, "name": "Montrose"},
    "77007": {"lat": 29.7751, "lon": -95.4011, "name": "Heights"},
    "77008": {"lat": 29.7994, "lon": -95.4156, "name": "Oak Forest"},
    "77009": {"lat": 29.7858, "lon": -95.3660, "name": "Northside"},
    "77010": {"lat": 29.7548, "lon": -95.3540, "name": "Downtown"},
    "77011": {"lat": 29.7372, "lon": -95.2958, "name": "Harrisburg"},
    "77012": {"lat": 29.7189, "lon": -95.2685, "name": "Gulfgate"},
    "77013": {"lat": 29.7684, "lon": -95.2481, "name": "Jacinto City"},
    "77014": {"lat": 29.8547, "lon": -95.4689, "name": "Cypress Station"},
    "77015": {"lat": 29.7775, "lon": -95.1887, "name": "Channelview"},
    "77016": {"lat": 29.8326, "lon": -95.3104, "name": "Kashmere Gardens"},
    "77017": {"lat": 29.6978, "lon": -95.2559, "name": "Park Place"},
    "77018": {"lat": 29.8214, "lon": -95.4348, "name": "Oak Forest North"},
    "77019": {"lat": 29.7559, "lon": -95.4147, "name": "River Oaks"},
    "77020": {"lat": 29.7758, "lon": -95.3066, "name": "Fifth Ward"},
    "77021": {"lat": 29.7047, "lon": -95.3559, "name": "South MacGregor"},
    "77022": {"lat": 29.8301, "lon": -95.3575, "name": "Independence Heights"},
    "77023": {"lat": 29.7202, "lon": -95.3165, "name": "Eastwood"},
    "77024": {"lat": 29.7770, "lon": -95.4947, "name": "Memorial"},
    "77025": {"lat": 29.6897, "lon": -95.4331, "name": "Braeswood"},
    "77026": {"lat": 29.7914, "lon": -95.3340, "name": "Kashmere"},
    "77027": {"lat": 29.7380, "lon": -95.4359, "name": "Galleria"},
    "77028": {"lat": 29.8070, "lon": -95.2924, "name": "Denver Harbor"},
    "77029": {"lat": 29.7556, "lon": -95.2486, "name": "Galena Park"},
    "77030": {"lat": 29.7071, "lon": -95.3980, "name": "Medical Center"},
    "77031": {"lat": 29.6659, "lon": -95.5311, "name": "Sharpstown South"},
    "77032": {"lat": 29.8667, "lon": -95.3349, "name": "IAH Airport"},
    "77033": {"lat": 29.6689, "lon": -95.3199, "name": "South Park"},
    "77034": {"lat": 29.6225, "lon": -95.2074, "name": "Ellington"},
    "77035": {"lat": 29.6681, "lon": -95.4807, "name": "Meyerland"},
    "77036": {"lat": 29.6996, "lon": -95.5367, "name": "Sharpstown"},
    "77037": {"lat": 29.8661, "lon": -95.3880, "name": "Greenspoint"},
    "77038": {"lat": 29.8889, "lon": -95.4227, "name": "Greenspoint North"},
    "77039": {"lat": 29.8963, "lon": -95.3440, "name": "Acres Homes North"},
    "77040": {"lat": 29.8677, "lon": -95.5290, "name": "Northwest"},
    "77041": {"lat": 29.8492, "lon": -95.5816, "name": "Jersey Village"},
    "77042": {"lat": 29.7388, "lon": -95.5611, "name": "Westchase"},
    "77043": {"lat": 29.7832, "lon": -95.5562, "name": "Spring Branch West"},
    "77044": {"lat": 29.8486, "lon": -95.1883, "name": "Atascocita"},
    "77045": {"lat": 29.6358, "lon": -95.4247, "name": "Hiram Clarke"},
    "77046": {"lat": 29.7298, "lon": -95.4560, "name": "Uptown"},
    "77047": {"lat": 29.6158, "lon": -95.3668, "name": "Sunnyside"},
    "77048": {"lat": 29.6172, "lon": -95.3101, "name": "South Acres"},
    "77049": {"lat": 29.7933, "lon": -95.1319, "name": "Sheldon"},
    "77050": {"lat": 29.9047, "lon": -95.3227, "name": "North Houston"},
    "77051": {"lat": 29.6558, "lon": -95.3680, "name": "South Park"},
    "77053": {"lat": 29.5883, "lon": -95.4883, "name": "Fort Bend Houston"},
    "77054": {"lat": 29.6886, "lon": -95.3997, "name": "South Main"},
    "77055": {"lat": 29.7843, "lon": -95.4688, "name": "Spring Branch"},
    "77056": {"lat": 29.7388, "lon": -95.4711, "name": "Galleria Area"},
    "77057": {"lat": 29.7489, "lon": -95.4911, "name": "Tanglewood"},
    "77058": {"lat": 29.5539, "lon": -95.0969, "name": "Clear Lake"},
    "77059": {"lat": 29.6014, "lon": -95.1275, "name": "Clear Lake City"},
    "77060": {"lat": 29.8994, "lon": -95.4069, "name": "North Houston"},
    "77061": {"lat": 29.6561, "lon": -95.2644, "name": "Gulfgate"},
    "77062": {"lat": 29.5775, "lon": -95.1208, "name": "Nassau Bay"},
    "77063": {"lat": 29.7206, "lon": -95.5145, "name": "Sharpstown North"},
    "77064": {"lat": 29.9092, "lon": -95.5206, "name": "Willowbrook"},
    "77065": {"lat": 29.9289, "lon": -95.5725, "name": "Cypress"},
    "77066": {"lat": 29.9283, "lon": -95.4722, "name": "Champions"},
    "77067": {"lat": 29.9081, "lon": -95.4347, "name": "Greenspoint"},
    "77068": {"lat": 29.9456, "lon": -95.4656, "name": "Champions"},
    "77069": {"lat": 29.9561, "lon": -95.5022, "name": "Cypress"},
    "77070": {"lat": 29.9719, "lon": -95.5400, "name": "Cypress"},
    "77071": {"lat": 29.6461, "lon": -95.5189, "name": "Fondren Southwest"},
    "77072": {"lat": 29.6994, "lon": -95.5933, "name": "Alief"},
    "77073": {"lat": 29.9461, "lon": -95.3867, "name": "North Houston"},
    "77074": {"lat": 29.6858, "lon": -95.5197, "name": "Gulfton"},
    "77075": {"lat": 29.6242, "lon": -95.2500, "name": "South Houston"},
    "77076": {"lat": 29.8611, "lon": -95.3461, "name": "Acres Homes"},
    "77077": {"lat": 29.7628, "lon": -95.5942, "name": "Energy Corridor"},
    "77078": {"lat": 29.8239, "lon": -95.2450, "name": "Cloverleaf"},
    "77079": {"lat": 29.7600, "lon": -95.6333, "name": "Memorial West"},
    "77080": {"lat": 29.8053, "lon": -95.5231, "name": "Spring Branch"},
    "77081": {"lat": 29.7139, "lon": -95.4864, "name": "Bellaire"},
    "77082": {"lat": 29.7347, "lon": -95.6267, "name": "Westheimer"},
    "77083": {"lat": 29.6894, "lon": -95.6497, "name": "Alief West"},
    "77084": {"lat": 29.8200, "lon": -95.6667, "name": "Katy"},
    "77085": {"lat": 29.6214, "lon": -95.4750, "name": "Fondren Park"},
    "77086": {"lat": 29.9031, "lon": -95.4583, "name": "Champions"},
    "77087": {"lat": 29.6847, "lon": -95.2936, "name": "Gulfgate"},
    "77088": {"lat": 29.8706, "lon": -95.3917, "name": "Acres Homes"},
    "77089": {"lat": 29.5878, "lon": -95.2256, "name": "South Belt"},
    "77090": {"lat": 29.9500, "lon": -95.4167, "name": "Champions"},
    "77091": {"lat": 29.8494, "lon": -95.4083, "name": "Oak Forest North"},
    "77092": {"lat": 29.8303, "lon": -95.4583, "name": "Oak Forest"},
    "77093": {"lat": 29.8486, "lon": -95.3261, "name": "Aldine"},
    "77094": {"lat": 29.7831, "lon": -95.7008, "name": "Energy Corridor West"},
    "77095": {"lat": 29.8831, "lon": -95.6458, "name": "Copperfield"},
    "77096": {"lat": 29.6656, "lon": -95.4558, "name": "Meyerland"},
    "77098": {"lat": 29.7356, "lon": -95.4053, "name": "Upper Kirby"},
    "77099": {"lat": 29.6739, "lon": -95.5881, "name": "Alief"},
    # Surrounding Areas (Fort Bend, Montgomery, Galveston, Brazoria, etc.)
    "77301": {"lat": 30.3080, "lon": -95.4551, "name": "Conroe"},
    "77339": {"lat": 30.0506, "lon": -95.0894, "name": "Kingwood"},
    "77345": {"lat": 30.0133, "lon": -95.1539, "name": "Kingwood"},
    "77346": {"lat": 30.0167, "lon": -95.1500, "name": "Humble"},
    "77354": {"lat": 30.1500, "lon": -95.5333, "name": "Magnolia"},
    "77355": {"lat": 30.0833, "lon": -95.6667, "name": "Magnolia"},
    "77356": {"lat": 30.3167, "lon": -95.6000, "name": "Montgomery"},
    "77357": {"lat": 30.1333, "lon": -95.3333, "name": "New Caney"},
    "77362": {"lat": 30.2000, "lon": -95.6333, "name": "Pinehurst"},
    "77365": {"lat": 30.1167, "lon": -95.2500, "name": "Porter"},
    "77373": {"lat": 30.0333, "lon": -95.4167, "name": "Spring"},
    "77375": {"lat": 30.0833, "lon": -95.5333, "name": "Tomball"},
    "77377": {"lat": 30.0667, "lon": -95.6167, "name": "Tomball"},
    "77378": {"lat": 30.3833, "lon": -95.3833, "name": "Willis"},
    "77379": {"lat": 30.0333, "lon": -95.4667, "name": "Spring"},
    "77380": {"lat": 30.1667, "lon": -95.4667, "name": "The Woodlands"},
    "77381": {"lat": 30.1667, "lon": -95.5000, "name": "The Woodlands"},
    "77382": {"lat": 30.2000, "lon": -95.5333, "name": "The Woodlands"},
    "77384": {"lat": 30.2167, "lon": -95.4500, "name": "Conroe"},
    "77385": {"lat": 30.1833, "lon": -95.4167, "name": "Conroe"},
    "77386": {"lat": 30.1167, "lon": -95.4000, "name": "Spring"},
    "77388": {"lat": 30.0667, "lon": -95.4500, "name": "Spring"},
    "77389": {"lat": 30.1000, "lon": -95.4833, "name": "Spring"},
    # Fort Bend County
    "77406": {"lat": 29.6667, "lon": -95.8500, "name": "Richmond"},
    "77407": {"lat": 29.6833, "lon": -95.7500, "name": "Richmond"},
    "77429": {"lat": 29.9500, "lon": -95.6667, "name": "Cypress"},
    "77433": {"lat": 29.9167, "lon": -95.7333, "name": "Cypress"},
    "77449": {"lat": 29.8167, "lon": -95.7500, "name": "Katy"},
    "77450": {"lat": 29.7667, "lon": -95.7667, "name": "Katy"},
    "77459": {"lat": 29.5167, "lon": -95.6333, "name": "Missouri City"},
    "77461": {"lat": 29.4000, "lon": -95.7667, "name": "Needville"},
    "77469": {"lat": 29.6000, "lon": -95.7667, "name": "Richmond"},
    "77471": {"lat": 29.5833, "lon": -95.8500, "name": "Rosenberg"},
    "77477": {"lat": 29.6167, "lon": -95.5333, "name": "Stafford"},
    "77478": {"lat": 29.5833, "lon": -95.5500, "name": "Sugar Land"},
    "77479": {"lat": 29.5500, "lon": -95.6000, "name": "Sugar Land"},
    "77484": {"lat": 30.0333, "lon": -95.8500, "name": "Waller"},
    "77485": {"lat": 29.4833, "lon": -95.9333, "name": "Wallis"},
    "77486": {"lat": 29.2833, "lon": -95.9500, "name": "West Columbia"},
    "77489": {"lat": 29.5500, "lon": -95.5167, "name": "Missouri City"},
    "77494": {"lat": 29.7833, "lon": -95.8167, "name": "Katy"},
    "77498": {"lat": 29.5667, "lon": -95.5667, "name": "Sugar Land"},
    # Galveston County
    "77510": {"lat": 29.3833, "lon": -95.0500, "name": "Santa Fe"},
    "77511": {"lat": 29.4000, "lon": -95.2333, "name": "Alvin"},
    "77517": {"lat": 29.4167, "lon": -95.1333, "name": "Santa Fe"},
    "77518": {"lat": 29.5167, "lon": -94.9833, "name": "Bacliff"},
    "77520": {"lat": 29.5333, "lon": -94.9500, "name": "Baytown"},
    "77521": {"lat": 29.7500, "lon": -94.9667, "name": "Baytown"},
    "77530": {"lat": 29.7500, "lon": -95.1500, "name": "Channelview"},
    "77534": {"lat": 29.3667, "lon": -95.4167, "name": "Danbury"},
    "77536": {"lat": 29.6333, "lon": -95.0500, "name": "Deer Park"},
    "77539": {"lat": 29.5167, "lon": -95.0500, "name": "Dickinson"},
    "77546": {"lat": 29.5000, "lon": -95.1833, "name": "Friendswood"},
    "77550": {"lat": 29.3000, "lon": -94.7833, "name": "Galveston"},
    "77551": {"lat": 29.2833, "lon": -94.8333, "name": "Galveston"},
    "77554": {"lat": 29.2000, "lon": -94.9500, "name": "Galveston West"},
    "77563": {"lat": 29.4667, "lon": -95.0000, "name": "Hitchcock"},
    "77565": {"lat": 29.5000, "lon": -94.9333, "name": "Kemah"},
    "77568": {"lat": 29.5167, "lon": -94.9000, "name": "La Marque"},
    "77571": {"lat": 29.6000, "lon": -95.0833, "name": "La Porte"},
    "77573": {"lat": 29.5333, "lon": -95.0833, "name": "League City"},
    "77581": {"lat": 29.4333, "lon": -95.2500, "name": "Pearland"},
    "77584": {"lat": 29.5000, "lon": -95.2833, "name": "Pearland"},
    "77586": {"lat": 29.5500, "lon": -95.0000, "name": "Seabrook"},
    "77590": {"lat": 29.4000, "lon": -94.9167, "name": "Texas City"},
    "77591": {"lat": 29.4333, "lon": -94.9500, "name": "Texas City"},
    # Brazoria County (South Houston)
    "77515": {"lat": 29.2000, "lon": -95.4333, "name": "Angleton"},
    "77531": {"lat": 29.0333, "lon": -95.4333, "name": "Clute"},
    "77541": {"lat": 28.9833, "lon": -95.3500, "name": "Freeport"},
    "77566": {"lat": 29.0500, "lon": -95.4667, "name": "Lake Jackson"},
    "77578": {"lat": 29.3000, "lon": -95.3333, "name": "Manvel"},
    # Chambers/Jefferson County (East)
    "77514": {"lat": 29.7500, "lon": -94.6500, "name": "Anahuac"},
    "77523": {"lat": 29.6500, "lon": -94.8500, "name": "Baytown East"},
    "77562": {"lat": 29.7833, "lon": -94.8000, "name": "Highlands"},
    "77617": {"lat": 29.5667, "lon": -94.5000, "name": "Gilchrist"},
    "77623": {"lat": 29.5333, "lon": -94.3667, "name": "High Island"},
    "77630": {"lat": 30.0833, "lon": -94.1333, "name": "Orange"},
    "77640": {"lat": 29.8833, "lon": -93.9500, "name": "Port Arthur"},
    "77642": {"lat": 29.9000, "lon": -94.0333, "name": "Port Arthur"},
    "77651": {"lat": 29.9333, "lon": -94.1833, "name": "Port Neches"},
    "77468": {"lat": 29.4333, "lon": -95.8833, "name": "Pledger"},
    # Waller County
    "77423": {"lat": 29.9000, "lon": -95.8833, "name": "Brookshire"},
    "77445": {"lat": 30.0500, "lon": -95.9833, "name": "Hempstead"},
    "77447": {"lat": 30.0833, "lon": -95.9333, "name": "Hockley"},
    "77466": {"lat": 29.9333, "lon": -95.8167, "name": "Pattison"},
    "77493": {"lat": 29.7833, "lon": -95.8667, "name": "Katy West"},
    # Austin County (West)
    "77833": {"lat": 30.1667, "lon": -96.4000, "name": "Brenham"},
    "77835": {"lat": 30.0000, "lon": -96.4667, "name": "Burton"},
    "77836": {"lat": 30.4167, "lon": -96.6167, "name": "Caldwell"},
    # Matagorda County (Southwest)
    "77414": {"lat": 28.9833, "lon": -95.9667, "name": "Bay City"},
    "77420": {"lat": 29.1333, "lon": -95.8333, "name": "Blessing"},
    "77428": {"lat": 28.7500, "lon": -96.0167, "name": "Caney"},
    "77440": {"lat": 28.9000, "lon": -96.1833, "name": "Elmaton"},
    "77465": {"lat": 28.7833, "lon": -96.0500, "name": "Palacios"},
    "77482": {"lat": 28.9833, "lon": -95.7833, "name": "Van Vleck"},
    "77483": {"lat": 28.7000, "lon": -96.1000, "name": "Wadsworth"},
    # Corpus Christi area (for reference)
    "78401": {"lat": 27.7969, "lon": -97.3964, "name": "Corpus Christi Downtown"},
    "78412": {"lat": 27.7167, "lon": -97.3667, "name": "Corpus Christi South"},
}


def compute_neighbors(zip_centroids, distance_km=15):
    """
    Compute neighboring ZIPs within distance_km of each other.
    Uses Haversine formula for distance calculation.
    """
    import math

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in km
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    neighbors = {}
    zips = list(zip_centroids.keys())

    for z1 in zips:
        neighbors[z1] = []
        lat1, lon1 = zip_centroids[z1]["lat"], zip_centroids[z1]["lon"]

        for z2 in zips:
            if z1 == z2:
                continue
            lat2, lon2 = zip_centroids[z2]["lat"], zip_centroids[z2]["lon"]
            dist = haversine(lat1, lon1, lat2, lon2)

            if dist <= distance_km:
                neighbors[z1].append(z2)

    return neighbors


def main():
    output_path = Path("data/processed/zip_coordinates.json")

    # Compute neighbors (ZIPs within 15km of each other)
    neighbors = compute_neighbors(HOUSTON_ZIP_CENTROIDS, distance_km=15)

    # Build output structure
    output = {
        "description": "Harris County and Greater Houston ZIP code centroids with neighbors",
        "distance_km_for_neighbors": 15,
        "total_zips": len(HOUSTON_ZIP_CENTROIDS),
        "zip_codes": {},
    }

    for zip_code, data in HOUSTON_ZIP_CENTROIDS.items():
        output["zip_codes"][zip_code] = {
            "lat": data["lat"],
            "lon": data["lon"],
            "name": data["name"],
            "neighbors": neighbors.get(zip_code, []),
        }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))

    print(f"✓ Wrote {len(HOUSTON_ZIP_CENTROIDS)} ZIP codes to {output_path}")

    # Summary stats
    neighbor_counts = [len(v) for v in neighbors.values()]
    print(f"  Avg neighbors per ZIP: {sum(neighbor_counts)/len(neighbor_counts):.1f}")
    print(f"  Max neighbors: {max(neighbor_counts)}")
    print(f"  ZIPs with no neighbors: {sum(1 for c in neighbor_counts if c == 0)}")


if __name__ == "__main__":
    main()
