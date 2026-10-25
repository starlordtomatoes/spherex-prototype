# spherex explorer
# how to do: search archive, download cutouts, explore the sky, compare observations
# streamlit handles the interface

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

from io import BytesIO
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import units as u
from astroquery.ipac.irsa import Irsa
from scipy.ndimage import map_coordinates


# page setup

st.set_page_config(
    page_title="SPHEREx Explorer",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #090d18;
        color: #e6eaf2;
    }

    h1, h2, h3 {
        color: #c9d7ff;
    }

    div[data-testid="stMetric"] {
        background-color: #141b2d;
        border: 1px solid #293653;
        padding: 12px;
        border-radius: 8px;
    }

    div[data-testid="stSidebar"] {
        background-color: #101626;
    }

    .stButton button {
        background-color: #5267d9;
        color: white;
        border-radius: 6px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# default settings

if "ra_input" not in st.session_state:
    st.session_state["ra_input"] = 304.693508808

if "dec_input" not in st.session_state:
    st.session_state["dec_input"] = 42.4436872991

if "radius_input" not in st.session_state:
    st.session_state["radius_input"] = 1.0

if "size_input" not in st.session_state:
    st.session_state["size_input"] = 0.05


# search archive

def search_observations(ra, dec, radius):
    # convert coordinates, query irsa, return results

    coord = SkyCoord(ra, dec, unit="deg")

    results = Irsa.query_sia(
        pos=(coord, radius * u.arcsec),
        collection="spherex_qr2"
    )

    return results.to_pandas()


# download cutout

def download_cutout(access_url, ra, dec, size):
    # request a smaller fits image around the coordinates

    cutout_url = (
        f"{access_url}"
        f"?center={ra},{dec}d"
        f"&size={size}"
    )

    response = requests.get(cutout_url, timeout=120)
    response.raise_for_status()

    return response.content


# find archive columns

def find_column(dataframe, search_text):
    # find a column by part of its name

    for column in dataframe.columns:
        if search_text.lower() in str(column).lower():
            return column

    return None


# observation labels

def get_observation_label(row):
    # make archive rows easier to identify

    parts = []

    for name in ["band", "obs_id", "mjd"]:
        column = find_column(pd.DataFrame([row]), name)

        if column is not None and pd.notna(row[column]):
            parts.append(f"{name}: {row[column]}")

    if not parts:
        return "observation"

    return " | ".join(parts)


# get a fits image

def get_image_data(hdul, product):
    # get a data product and turn cubes into 2d images

    data = hdul[product].data

    if data is None:
        raise ValueError(f"no data in {product}")

    data = np.asarray(data, dtype=np.float64)

    if data.ndim > 2:
        data = data[data.shape[0] // 2]

    if data.ndim != 2:
        raise ValueError(f"{product} is not a 2d image")

    return data.copy()


# get a product if it exists

def get_optional_product(hdul, product):
    if product not in hdul or hdul[product].data is None:
        return None

    try:
        return get_image_data(hdul, product)
    except Exception:
        return None


# read fits data

def read_fits(cutout_bytes):
    # copy the data before closing the fits file

    with fits.open(BytesIO(cutout_bytes)) as hdul:
        image = get_image_data(hdul, "IMAGE")

        variance = get_optional_product(hdul, "VARIANCE")
        flags = get_optional_product(hdul, "FLAGS")

        header = hdul["IMAGE"].header.copy()

        try:
            image_wcs = WCS(header).celestial

            if not image_wcs.has_celestial:
                image_wcs = None

        except Exception:
            image_wcs = None

        extension_info = []

        for index, hdu in enumerate(hdul):
            extension_info.append({
                "extension": index,
                "name": hdu.name,
                "type": type(hdu).__name__,
                "shape": (
                    str(hdu.data.shape)
                    if hdu.data is not None
                    else "no data"
                )
            })

    return {
        "image": image,
        "variance": variance,
        "flags": flags,
        "header": header,
        "wcs": image_wcs,
        "extensions": pd.DataFrame(extension_info)
    }


# display an image

def plot_image(data, contrast, colour_map, title):
    # use percentiles so faint details are easier to see

    pixels = data[np.isfinite(data)]

    if len(pixels) == 0:
        raise ValueError("no finite pixel values found")

    low = np.percentile(pixels, contrast)
    high = np.percentile(pixels, 100 - contrast)

    if low >= high:
        low = np.min(pixels)
        high = np.max(pixels)

    if low >= high:
        low -= 0.5
        high += 0.5

    fig, ax = plt.subplots(figsize=(8, 7))

    image = ax.imshow(
        data,
        origin="lower",
        cmap=colour_map,
        vmin=low,
        vmax=high
    )

    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")
    ax.set_title(title)

    fig.colorbar(image, ax=ax, label="pixel value")
    fig.tight_layout()

    return fig


# move around the sky

def move_sky(direction):
    # move by 75% of the cutout width so neighbouring views overlap

    ra = st.session_state["ra_input"]
    dec = st.session_state["dec_input"]
    size = st.session_state["size_input"]

    distance = size * 0.75

    if direction == "left":
        ra -= distance / max(abs(np.cos(np.radians(dec))), 0.01)

    elif direction == "right":
        ra += distance / max(abs(np.cos(np.radians(dec))), 0.01)

    elif direction == "up":
        dec += distance

    elif direction == "down":
        dec -= distance

    st.session_state["ra_input"] = float(ra % 360)
    st.session_state["dec_input"] = float(np.clip(dec, -90, 90))

    st.session_state["search_ra"] = st.session_state["ra_input"]
    st.session_state["search_dec"] = st.session_state["dec_input"]

    # clear the old cutout and load the new location

    st.session_state.pop("cutout_bytes", None)
    st.session_state.pop("loaded_observation", None)

    st.session_state["auto_load_cutout"] = True

# align one image with another using sky coordinates

def align_image(source_data, source_wcs, reference_shape, reference_wcs, order=1):
    # map each reference pixel to the matching source pixel

    if source_wcs is None or reference_wcs is None:
        raise ValueError("the image is missing usable sky coordinates (WCS)")

    height, width = reference_shape

    y, x = np.indices((height, width))

    sky_coords = reference_wcs.pixel_to_world(x, y)

    source_x, source_y = source_wcs.world_to_pixel(sky_coords)

    aligned = np.full(reference_shape, np.nan, dtype=np.float64)

    valid = (
        np.isfinite(source_x)
        & np.isfinite(source_y)
        & (source_x >= 0)
        & (source_y >= 0)
        & (source_x < source_data.shape[1] - 1)
        & (source_y < source_data.shape[0] - 1)
    )

    if np.any(valid):
        aligned[valid] = map_coordinates(
            source_data,
            [source_y[valid], source_x[valid]],
            order=order,
            mode="constant",
            cval=np.nan
        )

    return aligned


# compare two observations

def compare_images(earlier_bytes, later_bytes, threshold):
    # align the later image to the earlier image before comparing pixels

    earlier = read_fits(earlier_bytes)
    later = read_fits(later_bytes)

    earlier_image = earlier["image"]
    later_image = later["image"]

    if earlier["wcs"] is None or later["wcs"] is None:
        raise ValueError(
            "one of these images has no usable WCS. "
            "the images cannot be reliably aligned."
        )

    later_aligned = align_image(
        later_image,
        later["wcs"],
        earlier_image.shape,
        earlier["wcs"]
    )

    earlier_variance = earlier["variance"]
    later_variance = later["variance"]

    # align variance maps if both observations provide them

    combined_variance = None

    if earlier_variance is not None and later_variance is not None:
        if earlier_variance.shape != earlier_image.shape:
            earlier_variance = None

        if later_variance.shape != later_image.shape:
            later_variance = None

        if earlier_variance is not None and later_variance is not None:
            later_variance_aligned = align_image(
                later_variance,
                later["wcs"],
                earlier_image.shape,
                earlier["wcs"]
            )

            combined_variance = earlier_variance + later_variance_aligned

    # exclude pixels with non-zero quality flags when flags are available

    valid = (
        np.isfinite(earlier_image)
        & np.isfinite(later_aligned)
    )

    if earlier["flags"] is not None:
        if earlier["flags"].shape == earlier_image.shape:
            valid &= earlier["flags"] == 0

    if later["flags"] is not None:
        if later["flags"].shape == later_image.shape:
            later_flags_aligned = align_image(
                later["flags"],
                later["wcs"],
                earlier_image.shape,
                earlier["wcs"],
                order=0
            )

            valid &= np.isfinite(later_flags_aligned)
            valid &= later_flags_aligned == 0

    if np.count_nonzero(valid) < 10:
        raise ValueError(
            "too few valid overlapping pixels to compare these observations."
        )

    # calculate the brightness difference

    difference = later_aligned - earlier_image

    difference[~valid] = np.nan

    # estimate how significant the differences are

    significance = np.full(earlier_image.shape, np.nan)

    if combined_variance is not None:
        usable_variance = (
            valid
            & np.isfinite(combined_variance)
            & (combined_variance > 0)
        )

        significance[usable_variance] = (
            difference[usable_variance]
            / np.sqrt(combined_variance[usable_variance])
        )

        significance_method = "combined variance maps"

    else:
        # fallback when the archive doesn't provide usable variance maps

        values = difference[valid]

        median_difference = np.median(values)

        noise = 1.4826 * np.median(
            np.abs(values - median_difference)
        )

        if not np.isfinite(noise) or noise <= 0:
            noise = np.std(values)

        if not np.isfinite(noise) or noise <= 0:
            raise ValueError(
                "could not estimate the noise in the difference image."
            )

        significance[valid] = (
            difference[valid] - median_difference
        ) / noise

        significance_method = "robust noise estimate"

    # identify unusually large differences

    candidates = valid & (
        np.abs(significance) >= threshold
    )

    # remove candidates from invalid pixels

    significance[~valid] = np.nan

    return {
        "earlier": earlier,
        "later": later,
        "later_aligned": later_aligned,
        "difference": difference,
        "significance": significance,
        "candidates": candidates,
        "valid": valid,
        "method": significance_method
    }


# title

st.title("SPHEREx Explorer")

st.write(
    "search the sky, load infrared cutouts, "
    "and compare observations for possible changes."
)

st.caption("NASA SPHEREx public archive through IRSA")


# search controls

st.sidebar.header("search the sky")

ra = st.sidebar.number_input(
    "right ascension (degrees)",
    min_value=0.0,
    max_value=360.0,
    value=304.693508808,
    format="%.8f",
    key="ra_input"
)

dec = st.sidebar.number_input(
    "declination (degrees)",
    min_value=-90.0,
    max_value=90.0,
    value=42.4436872991,
    format="%.8f",
    key="dec_input"
)

radius = st.sidebar.number_input(
    "search radius (arcseconds)",
    min_value=0.1,
    max_value=3600.0,
    value=1.0,
    step=0.5,
    key="radius_input"
)

cutout_size = st.sidebar.number_input(
    "cutout size (degrees)",
    min_value=0.005,
    max_value=1.0,
    value=0.05,
    step=0.005,
    format="%.3f",
    key="size_input"
)

search_button = st.sidebar.button(
    "search observations",
    use_container_width=True
)


# run search

if search_button:
    st.session_state["search_ra"] = ra
    st.session_state["search_dec"] = dec
    st.session_state["search_radius"] = radius
    st.session_state["cutout_size"] = cutout_size

    st.session_state.pop("cutout_bytes", None)
    st.session_state.pop("loaded_observation", None)
    st.session_state.pop("search_error", None)
    st.session_state["auto_load_cutout"] = False

    with st.spinner("searching archive..."):
        try:
            observations = search_observations(
                ra,
                dec,
                radius
            )

            st.session_state["observations"] = observations

        except Exception as error:
            st.session_state.pop("observations", None)
            st.session_state["search_error"] = str(error)


# show search errors

if "search_error" in st.session_state:
    st.error(
        f"archive search failed: {st.session_state['search_error']}"
    )


# show results

if "observations" not in st.session_state:
    st.info("search for observations using the sidebar.")

else:
    observations = st.session_state["observations"]

    if observations.empty:
        st.warning("no observations found.")

    else:
        # overview

        st.success(f"found {len(observations)} observations")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "observations",
            len(observations)
        )

        col2.metric(
            "RA",
            f"{st.session_state['search_ra']:.5f}°"
        )

        col3.metric(
            "Dec",
            f"{st.session_state['search_dec']:.5f}°"
        )

        st.subheader("observations")

        st.dataframe(
            observations,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # choose observation

        st.subheader("load an image")

        access_column = find_column(
            observations,
            "access_url"
        )

        if access_column is None:
            st.error("no access_url column found.")

        else:
            if "selected_index" not in st.session_state:
                st.session_state["selected_index"] = 0

            # keep the selection valid if a new search returns fewer rows

            if st.session_state["selected_index"] >= len(observations):
                st.session_state["selected_index"] = 0

            selected_index = st.selectbox(
                "observation",
                options=list(range(len(observations))),
                format_func=lambda index: (
                    f"{index + 1}. "
                    f"{get_observation_label(observations.iloc[index])}"
                ),
                key="selected_index"
            )

            access_url = observations.iloc[selected_index][access_column]

            # download automatically after moving around the sky

            auto_load = st.session_state.pop(
                "auto_load_cutout",
                False
            )

            if st.session_state.get("loaded_observation") != selected_index:
                st.session_state.pop("cutout_bytes", None)

            should_load = st.button("load image cutout") or auto_load

            if should_load:
                with st.spinner("downloading cutout..."):
                    try:
                        cutout_bytes = download_cutout(
                            access_url,
                            st.session_state["search_ra"],
                            st.session_state["search_dec"],
                            st.session_state.get(
                                "cutout_size",
                                cutout_size
                            )
                        )

                        st.session_state["cutout_bytes"] = cutout_bytes
                        st.session_state["loaded_observation"] = selected_index

                    except Exception as error:
                        st.error(f"cutout download failed: {error}")

            # view fits data

            if (
                "cutout_bytes" in st.session_state
                and st.session_state.get("loaded_observation") == selected_index
            ):
                try:
                    fits_data = read_fits(
                        st.session_state["cutout_bytes"]
                    )

                    products = []

                    with fits.open(
                        BytesIO(st.session_state["cutout_bytes"])
                    ) as hdul:
                        products = [
                            name
                            for name in [
                                "IMAGE",
                                "FLAGS",
                                "VARIANCE",
                                "ZODI"
                            ]
                            if name in hdul and hdul[name].data is not None
                        ]

                    if not products:
                        st.error("no supported image products found.")

                    else:
                        st.subheader("image viewer")

                        col1, col2 = st.columns([3, 1])

                        with col2:
                            product = st.selectbox(
                                "data",
                                options=products,
                                index=(
                                    products.index("IMAGE")
                                    if "IMAGE" in products
                                    else 0
                                )
                            )

                            colour_map = st.selectbox(
                                "colour map",
                                options=[
                                    "magma",
                                    "inferno",
                                    "viridis",
                                    "plasma",
                                    "gray",
                                    "coolwarm"
                                ]
                            )

                            contrast = st.slider(
                                "contrast",
                                min_value=0,
                                max_value=20,
                                value=2
                            )

                        if product == "IMAGE":
                            data = fits_data["image"]

                        elif product == "VARIANCE":
                            data = fits_data["variance"]

                        elif product == "FLAGS":
                            data = fits_data["flags"]

                        else:
                            with fits.open(
                                BytesIO(st.session_state["cutout_bytes"])
                            ) as hdul:
                                data = get_image_data(hdul, product)

                        if data is not None:
                            with col1:
                                fig = plot_image(
                                    data,
                                    contrast,
                                    colour_map,
                                    f"SPHEREx {product}"
                                )

                                st.pyplot(fig)
                                plt.close(fig)

                            # pixel stats

                            pixels = data[np.isfinite(data)]

                            st.subheader("pixel statistics")

                            stat1, stat2, stat3, stat4 = st.columns(4)

                            stat1.metric("pixels", f"{data.size:,}")

                            if len(pixels) > 0:
                                stat2.metric("minimum", f"{np.min(pixels):.5g}")
                                stat3.metric("maximum", f"{np.max(pixels):.5g}")
                                stat4.metric("mean", f"{np.mean(pixels):.5g}")

                            if product == "FLAGS":
                                st.caption(
                                    "quality flags, not image brightness."
                                )

                            elif product == "VARIANCE":
                                st.caption(
                                    "measurement variance, not image brightness."
                                )

                            elif product == "ZODI":
                                st.caption(
                                    "zodiacal light data, if provided by this product."
                                )

                        # download fits file

                        st.download_button(
                            "download fits",
                            data=st.session_state["cutout_bytes"],
                            file_name="spherex_cutout.fits",
                            mime="application/fits"
                        )

                        # fits extensions

                        with st.expander("fits file information"):
                            st.dataframe(
                                fits_data["extensions"],
                                use_container_width=True,
                                hide_index=True
                            )

                except Exception as error:
                    st.error(f"could not read fits data: {error}")

            st.divider()

            # sky navigation

            st.subheader("explore neighbouring areas")

            st.write(
                "move the cutout centre to a nearby part of the sky. "
                "the new cutout will load automatically."
            )

            left, middle, right = st.columns([1, 1, 1])

            with middle:
                st.button(
                    "↑ north",
                    use_container_width=True,
                    on_click=move_sky,
                    args=("up",)
                )

            with left:
                st.button(
                    "← east",
                    use_container_width=True,
                    on_click=move_sky,
                    args=("left",)
                )

            with middle:
                st.button(
                    "↓ south",
                    use_container_width=True,
                    on_click=move_sky,
                    args=("down",)
                )

            with right:
                st.button(
                    "west →",
                    use_container_width=True,
                    on_click=move_sky,
                    args=("right",)
                )

            st.caption(
                "Each move covers 75% of the cutout width. "
                "The overlap helps avoid gaps between neighbouring views."
            )

            st.divider()

            # compare observations

            st.subheader("look for changes over time")

            st.write(
                "select two observations of the same region. "
                "the later image will be aligned with the earlier image "
                "before their pixel values are compared."
            )

            if len(observations) < 2:
                st.info(
                    "at least two observations are needed for comparison."
                )

            else:
                compare_col1, compare_col2 = st.columns(2)

                with compare_col1:
                    earlier_index = st.selectbox(
                        "earlier observation",
                        options=list(range(len(observations))),
                        format_func=lambda index: (
                            f"{index + 1}. "
                            f"{get_observation_label(observations.iloc[index])}"
                        ),
                        index=0,
                        key="earlier_index"
                    )

                with compare_col2:
                    later_index = st.selectbox(
                        "later observation",
                        options=list(range(len(observations))),
                        format_func=lambda index: (
                            f"{index + 1}. "
                            f"{get_observation_label(observations.iloc[index])}"
                        ),
                        index=min(1, len(observations) - 1),
                        key="later_index"
                    )

                threshold = st.slider(
                    "candidate significance threshold",
                    min_value=3.0,
                    max_value=8.0,
                    value=5.0,
                    step=0.5,
                    help=(
                        "Higher thresholds flag fewer pixels. "
                        "This is a screening threshold, not proof of a discovery."
                    )
                )

                if st.button("compare observations"):
                    if earlier_index == later_index:
                        st.error(
                            "choose two different observations."
                        )

                    else:
                        earlier_url = observations.iloc[
                            earlier_index
                        ][access_column]

                        later_url = observations.iloc[
                            later_index
                        ][access_column]

                        with st.spinner(
                            "downloading and aligning observations..."
                        ):
                            try:
                                earlier_bytes = download_cutout(
                                    earlier_url,
                                    st.session_state["search_ra"],
                                    st.session_state["search_dec"],
                                    st.session_state.get(
                                        "cutout_size",
                                        cutout_size
                                    )
                                )

                                later_bytes = download_cutout(
                                    later_url,
                                    st.session_state["search_ra"],
                                    st.session_state["search_dec"],
                                    st.session_state.get(
                                        "cutout_size",
                                        cutout_size
                                    )
                                )

                                result = compare_images(
                                    earlier_bytes,
                                    later_bytes,
                                    threshold
                                )

                                st.session_state["comparison_result"] = result
                                st.session_state["comparison_earlier"] = earlier_index
                                st.session_state["comparison_later"] = later_index
                                st.session_state["comparison_threshold"] = threshold

                            except Exception as error:
                                st.session_state.pop(
                                    "comparison_result",
                                    None
                                )

                                st.error(
                                    f"comparison failed: {error}"
                                )

                # display the saved comparison

                if "comparison_result" in st.session_state:
                    result = st.session_state["comparison_result"]

                    if (
                        st.session_state.get("comparison_earlier") == earlier_index
                        and st.session_state.get("comparison_later") == later_index
                    ):
                        st.caption(
                            f"significance method: {result['method']}"
                        )

                        valid_count = np.count_nonzero(result["valid"])
                        candidate_count = np.count_nonzero(result["candidates"])

                        metric1, metric2, metric3 = st.columns(3)

                        metric1.metric(
                            "valid overlapping pixels",
                            f"{valid_count:,}"
                        )

                        metric2.metric(
                            "candidate pixels",
                            f"{candidate_count:,}"
                        )

                        metric3.metric(
                            "threshold",
                            f"{threshold:.1f}σ"
                        )

                        # show the earlier, later, and difference images

                        earlier_image = result["earlier"]["image"]
                        later_image = result["later_aligned"]
                        difference = result["difference"]
                        significance = result["significance"]

                        image_col1, image_col2 = st.columns(2)

                        with image_col1:
                            fig = plot_image(
                                earlier_image,
                                2,
                                "magma",
                                "earlier observation"
                            )

                            st.pyplot(fig)
                            plt.close(fig)

                        with image_col2:
                            fig = plot_image(
                                later_image,
                                2,
                                "magma",
                                "later observation (aligned)"
                            )

                            st.pyplot(fig)
                            plt.close(fig)

                        diff_col1, diff_col2 = st.columns(2)

                        with diff_col1:
                            finite_diff = difference[
                                np.isfinite(difference)
                            ]

                            if len(finite_diff) > 0:
                                limit = np.percentile(
                                    np.abs(finite_diff),
                                    99
                                )

                                if limit <= 0:
                                    limit = 1.0

                                fig, ax = plt.subplots(figsize=(8, 7))

                                image = ax.imshow(
                                    difference,
                                    origin="lower",
                                    cmap="coolwarm",
                                    vmin=-limit,
                                    vmax=limit
                                )

                                ax.set_title("difference (later − earlier)")
                                ax.set_xlabel("pixel x")
                                ax.set_ylabel("pixel y")

                                fig.colorbar(
                                    image,
                                    ax=ax,
                                    label="change in pixel value"
                                )

                                fig.tight_layout()

                                st.pyplot(fig)
                                plt.close(fig)

                        with diff_col2:
                            fig, ax = plt.subplots(figsize=(8, 7))

                            image = ax.imshow(
                                significance,
                                origin="lower",
                                cmap="coolwarm",
                                vmin=-threshold,
                                vmax=threshold
                            )

                            # mark pixels that passed the threshold

                            candidate_y, candidate_x = np.where(
                                result["candidates"]
                            )

                            if len(candidate_x) > 0:
                                ax.scatter(
                                    candidate_x,
                                    candidate_y,
                                    s=12,
                                    facecolors="none",
                                    edgecolors="yellow",
                                    linewidths=0.7
                                )

                            ax.set_title("significance map and candidates")
                            ax.set_xlabel("pixel x")
                            ax.set_ylabel("pixel y")

                            fig.colorbar(
                                image,
                                ax=ax,
                                label="estimated significance"
                            )

                            fig.tight_layout()

                            st.pyplot(fig)
                            plt.close(fig)

                        st.warning(
                            "candidate pixels are not confirmed discoveries. "
                            "They may represent variable sources, transients, "
                            "noise, bad pixels or processing artefacts. "
                            "The observations still need scientific validation."
                        )

                        if result["method"] != "combined variance maps":
                            st.info(
                                "at least one usable variance map was missing. "
                                "Significance was estimated from the difference "
                                "image's overall noise, so the values are only "
                                "approximate and may not represent true statistical "
                                "significance."
                            )

                        # download the difference data

                        difference_buffer = BytesIO()

                        fits.writeto(
                            difference_buffer,
                            difference.astype(np.float32),
                            header=result["earlier"]["header"],
                            overwrite=True
                        )

                        st.download_button(
                            "download difference FITS",
                            data=difference_buffer.getvalue(),
                            file_name="spherex_difference.fits",
                            mime="application/fits"
                        )


# footer

st.divider()

st.caption(
    "experimental test project. unusual pixels do not automatically "
    "mean a new astronomical object. results depend on observation quality, "
    "alignment and the availability of repeated observations."
)