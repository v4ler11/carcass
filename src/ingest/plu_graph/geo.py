#!/usr/bin/env python3
"""End-to-end probe of the verified public APIs for the Gif-sur-Yvette task.

Resolves an address to a cadastral parcel, then pulls the zoning and overlay
attributes that the PLU PDFs encode graphically. Every endpoint here was
verified live; see docs/apis.md.

    uv run python -m ingest.plu_graph.geo
    uv run python -m ingest.plu_graph.geo --address "1 rue de la République 91190 Gif-sur-Yvette"
    uv run python -m ingest.plu_graph.geo --lat 48.709836 --lon 2.165254
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request

import numpy as np
from pyproj import Transformer

UA = {"User-Agent": "plu-graph/1.0 (PLU buildability assessment)"}
WGS84_TO_L93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
BAN_PLUS = "https://data.geopf.fr/wfs/ows"
APICARTO = "https://apicarto.ign.fr/api"
GPU = "https://www.geoportail-urbanisme.gouv.fr/api"


def get(url, params=None, timeout=60):
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def geocode(address):
    data = get("https://api-adresse.data.gouv.fr/search/", {"q": address, "limit": 1})
    if not data["features"]:
        raise SystemExit(f"BAN: no match for {address!r}")
    best = data["features"][0]
    lon, lat = best["geometry"]["coordinates"]
    return {
        "label": best["properties"]["label"],
        "score": best["properties"]["score"],
        "address_id": best["properties"]["id"],
        "insee": best["properties"]["citycode"],
        "lon": lon,
        "lat": lat,
    }


def parcel_for_point(lon, lat, pad=0.00015):
    """BAN-PLUS address<->parcel link. WFS 2.0 + EPSG:4326 takes lat,lon order."""
    data = get(
        BAN_PLUS,
        {
            "SERVICE": "WFS",
            "VERSION": "2.0.0",
            "REQUEST": "GetFeature",
            "TYPENAME": "BAN-PLUS:lien_adresse_parcelle",
            "SRSNAME": "EPSG:4326",
            "COUNT": 10,
            "OUTPUTFORMAT": "application/json",
            "BBOX": f"{lat - pad},{lon - pad},{lat + pad},{lon + pad}",
        },
    )
    links = [f["properties"] for f in data.get("features", [])]
    if not links:
        raise SystemExit("BAN-PLUS: no address/parcel link near the point")
    return links


def pick_link(links, address_id):
    for link in links:
        if link.get("id_adr") == address_id:
            return link
    return links[0]


def split_idu(idu):
    """idu is 14 chars: INSEE(5) + commune/abs(3) + section(2) + numero(4)."""
    return idu[:5], idu[5:8], idu[8:10], idu[10:14]


def parcel_geometry(idu):
    insee, _abs, section, numero = split_idu(idu)
    data = get(
        f"{APICARTO}/cadastre/parcelle",
        {"code_insee": insee, "section": section, "numero": numero},
    )
    feats = data.get("features", [])
    if not feats:
        raise SystemExit(f"cadastre: parcel {idu} not found")
    props = feats[0]["properties"]
    return {
        "idu": props["idu"],
        "section": props["section"],
        "numero": props["numero"],
        "area_m2": props["contenance"],
        "commune": props["nom_com"],
        "geometry": feats[0]["geometry"],
    }


def parcel_l93(idu):
    """``parcel_geometry`` with rings in Lambert-93, the frame the sheet is drawn in."""
    info = parcel_geometry(idu)
    geometry = info["geometry"]
    polygons = (
        geometry["coordinates"]
        if geometry["type"] == "MultiPolygon"
        else [geometry["coordinates"]]
    )
    rings = [
        np.array([WGS84_TO_L93.transform(x, y) for x, y in polygon[0]])
        for polygon in polygons
    ]
    return {**info, "rings_l93": rings}


def parcel_idu_for_address(address, pad=0.00015):
    """The cadastral idu the BAN address is linked to."""
    links = parcel_for_point(address["lon"], address["lat"], pad=pad)
    return pick_link(links, address["address_id"])["idu"]


def parcel_dossier(idu):
    """One-shot GPU feature-info: zoning + every prescription on the parcel."""
    insee, abs_, section, numero = split_idu(idu)
    dept, commune = insee[:2], insee[2:]
    pid = f"{dept}_{commune}_000_{abs_}_{section}_{numero}"
    du = get(
        f"{GPU}/feature-info/du",
        {
            "parcelId": pid,
            "typeName": "zone_urba,prescription_surf,prescription_lin,info_surf,info_lin,prescription_pct,info_pct",
        },
    )
    sup = get(
        f"{GPU}/feature-info/sup",
        {"parcelId": pid, "typeName": "assiette_sup_s,assiette_sup_l"},
    )
    return pid, du.get("features", []), sup.get("features", [])


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--address", default="4 rue Joliot Curie 91190 Gif-sur-Yvette")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    args = ap.parse_args(argv)

    if args.lat is not None and args.lon is not None:
        location = {
            "lat": args.lat,
            "lon": args.lon,
            "label": "(given point)",
            "address_id": None,
        }
    else:
        location = geocode(args.address)

    print(f"address        {location['label']}")
    print(f"coordinates    {location['lon']:.6f}, {location['lat']:.6f}  (WGS84)")
    if location["address_id"]:
        print(f"BAN id         {location['address_id']}")

    links = parcel_for_point(location["lon"], location["lat"])
    link = pick_link(links, location.get("address_id"))
    print(
        f"parcel (idu)   {link['idu']}   link={link.get('type_lien')} addresses={link.get('nb_adr')}"
    )

    parcel = parcel_geometry(link["idu"])
    print(
        f"parcel         {parcel['section']} {parcel['numero']}  {parcel['area_m2']:,} m2  {parcel['commune']}"
    )

    pid, du, sup = parcel_dossier(link["idu"])
    print(f"GPU parcel id  {pid}")

    zone = next((f for f in du if f["id"].startswith("zone_urba")), None)
    if zone:
        p = zone["properties"]
        print(f"zone           {p['libelle']}  type={p['typezone']}  doc={p['idurba']}")
    overlays = [f for f in du if not f["id"].startswith("zone_urba")]
    print(f"overlays       {len(overlays)}")
    for f in overlays:
        p = f["properties"]
        label = p.get("libelle") or ""
        tag = p.get("txt") or ""
        print(f"  - {f['id']:<28} {tag:<10} {label}")
    print(f"SUP            {len(sup)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
