import requests
from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

ra_deg = 304.693508808
dec_deg = 42.4436872991

coord = SkyCoord(ra_deg, dec_deg, unit="deg")
search_radius = 1 * u.arcsec

results = Irsa.query_sia(
    pos=(coord, search_radius),
    collection="spherex_qr2"
)

print("Number of results:", len(results))
print(results)

print("\nColumns:")
print(results.colnames)

print("\nFirst result:")
first = results[0]

# Request a small cutout from the original SPHEREx FITS file
cutout_url = (
    f"{first['access_url']}"
    f"?center={ra_deg},{dec_deg}&size=0.05"
)

print("\nRequesting cutout...")
response = requests.get(cutout_url, timeout=120)

print("HTTP status:", response.status_code)
print("Response size:", len(response.content), "bytes")
print("Content type:", response.headers.get("Content-Type"))

# Only save the file if the server returned a successful response
if response.status_code == 200 and len(response.content) > 1000:
    with open("spherex_cutout.fits", "wb") as file:
        file.write(response.content)

    print("SUCCESS! Saved spherex_cutout.fits")
else:
    print("Cutout request failed.")
    print(response.text[:1000])

# Open the downloaded FITS file
with fits.open("spherex_cutout.fits") as hdul:
    print("\nFITS file opened successfully!")
    hdul.info()

with fits.open("spherex_cutout.fits") as hdul:
    image = hdul["IMAGE"].data

    print("Image shape:", image.shape)
    print("Minimum:", image.min())
    print("Maximum:", image.max())

    plt.figure(figsize=(8, 8))
    valid_pixels = image[np.isfinite(image)]
    vmin, vmax = np.percentile(valid_pixels, [1, 99])

    plt.imshow(
        image,
        origin="lower",
        cmap="inferno",
        vmin=vmin,
        vmax=vmax
    )
    plt.colorbar(label="Pixel intensity")
    plt.title("SPHEREx Science Image")
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    plt.show()