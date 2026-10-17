"""
Gradio app to test YOUR Azure ML online endpoint for the Olist freight-cost model.

No credentials are stored in this file. The app reads them from environment variables, or
from a file called .env next to this script, which YOU create after deploying your endpoint:

    FREIGHT_ENDPOINT_URL=https://<your-endpoint>.<region>.inference.ml.azure.com/score
    FREIGHT_ENDPOINT_KEY=<your endpoint key>

Create .env automatically:   ..\scripts\create-app-env.ps1 -EndpointName <your-endpoint>
Run the app:                 python app.py      then open http://127.0.0.1:7860

You can also paste the URL and key into the "Connection" panel of the app; they are kept in
memory only for that browser session.
"""

import json
import math
import os
import time

import gradio as gr
import requests

try:  # python-dotenv is optional: without it, set the environment variables yourself
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

ENV_URL = os.environ.get("FREIGHT_ENDPOINT_URL", "").strip()
ENV_KEY = os.environ.get("FREIGHT_ENDPOINT_KEY", "").strip()

# -----------------------------------------------------------------------------------------
# Reference data derived from the Olist dataset (used only to pre-fill the distance field)
# -----------------------------------------------------------------------------------------
# Median coordinates of all zip codes in each Brazilian state (olist_geolocation_dataset.csv)
STATE_CENTROIDS = {
    "AC": (-9.96, -67.83), "AL": (-9.64, -35.75), "AM": (-3.09, -60.01), "AP": (0.03, -51.07),
    "BA": (-12.96, -38.96), "CE": (-3.79, -38.59), "DF": (-15.82, -47.94), "ES": (-20.29, -40.32),
    "GO": (-16.68, -49.27), "MA": (-3.29, -44.30), "MG": (-19.92, -43.98), "MS": (-20.48, -54.61),
    "MT": (-15.24, -56.04), "PA": (-1.46, -48.48), "PB": (-7.13, -35.49), "PE": (-8.08, -34.95),
    "PI": (-5.10, -42.76), "PR": (-25.38, -50.62), "RJ": (-22.87, -43.24), "RN": (-5.84, -35.26),
    "RO": (-10.43, -63.02), "RR": (2.82, -60.70), "RS": (-29.80, -51.25), "SC": (-27.10, -49.04),
    "SE": (-10.93, -37.07), "SP": (-23.53, -46.69), "TO": (-10.18, -48.33),
}
# Median seller-to-customer distance for shipments inside the same state (Olist orders);
# states without enough same-state orders fall back to DEFAULT_WITHIN_STATE_KM.
WITHIN_STATE_KM = {
    "BA": 101, "ES": 71, "GO": 51, "MG": 241, "MT": 412, "PE": 49, "PI": 165,
    "PR": 262, "RJ": 36, "RS": 107, "SC": 116, "SP": 95,
}
DEFAULT_WITHIN_STATE_KM = 50
STATES = sorted(STATE_CENTROIDS)

CATEGORIES = [
    "agro_industry_and_commerce", "air_conditioning", "art", "arts_and_craftmanship", "audio", "auto",
    "baby", "bed_bath_table", "books_general_interest", "books_imported", "books_technical",
    "cds_dvds_musicals", "christmas_supplies", "cine_photo", "computers", "computers_accessories",
    "consoles_games", "construction_tools_construction", "construction_tools_lights",
    "construction_tools_safety", "cool_stuff", "costruction_tools_garden", "costruction_tools_tools",
    "diapers_and_hygiene", "drinks", "dvds_blu_ray", "electronics", "fashio_female_clothing",
    "fashion_bags_accessories", "fashion_childrens_clothes", "fashion_male_clothing", "fashion_shoes",
    "fashion_sport", "fashion_underwear_beach", "fixed_telephony", "flowers", "food", "food_drink",
    "furniture_bedroom", "furniture_decor", "furniture_living_room", "furniture_mattress_and_upholstery",
    "garden_tools", "health_beauty", "home_appliances", "home_appliances_2", "home_comfort_2",
    "home_confort", "home_construction", "housewares", "industry_commerce_and_business",
    "kitchen_dining_laundry_garden_furniture", "la_cuisine", "luggage_accessories", "market_place",
    "music", "musical_instruments", "office_furniture", "party_supplies", "pc_gamer", "perfumery",
    "pet_shop", "portateis_cozinha_e_preparadores_de_alimentos", "security_and_services",
    "signaling_and_security", "small_appliances", "small_appliances_home_oven_and_coffee",
    "sports_leisure", "stationery", "tablets_printing_image", "telephony", "toys", "watches_gifts",
    "unknown",
]


def haversine_km(a, b):
    """Great-circle distance between two (lat, lng) points, in kilometres."""
    lat1, lon1, lat2, lon2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def estimate_distance(seller_state, customer_state):
    """Rough distance used to pre-fill the form. Students can overwrite it."""
    if not seller_state or not customer_state:
        return gr.update()
    if seller_state == customer_state:
        return WITHIN_STATE_KM.get(seller_state, DEFAULT_WITHIN_STATE_KM)
    return round(haversine_km(STATE_CENTROIDS[seller_state], STATE_CENTROIDS[customer_state]))


# -----------------------------------------------------------------------------------------
# Call the endpoint
# -----------------------------------------------------------------------------------------
def predict(seller_state, customer_state, category, price, weight_g, length_cm, height_cm, width_cm,
            distance_km, items_in_order, url_input, key_input):
    url = (url_input or ENV_URL).strip()
    key = (key_input or ENV_KEY).strip()
    if not url or not key:
        return ("### Not connected\nAdd your endpoint URL and key: create `app/.env` "
                "(see `.env.example`) or paste them into the **Connection** panel."), None, None

    numbers = {"price": price, "product_weight_g": weight_g, "product_length_cm": length_cm,
               "product_height_cm": height_cm, "product_width_cm": width_cm, "distance_km": distance_km}
    missing = [name for name, value in numbers.items() if value is None]
    if missing:
        return f"### Missing input\nPlease fill in: {', '.join(missing)}", None, None

    payload = {"input_data": [{
        "seller_state": seller_state,
        "customer_state": customer_state,
        "product_category": category or "unknown",
        **{name: float(value) for name, value in numbers.items()},
        "items_in_order": int(items_in_order or 1),
        "items_from_seller": int(items_in_order or 1),
    }]}

    started = time.perf_counter()
    try:
        response = requests.post(url, json=payload, headers={"Authorization": f"Bearer {key}"}, timeout=30)
    except requests.RequestException as error:
        return f"### Could not reach the endpoint\n`{error}`\n\nCheck the URL.", payload, None
    latency_ms = (time.perf_counter() - started) * 1000

    if response.status_code in (401, 403):
        return "### Key rejected (HTTP %d)\nCheck that the key belongs to this endpoint." % response.status_code, payload, None
    if response.status_code == 404:
        return "### Endpoint not found (HTTP 404)\nCheck the URL; it must end with `/score`.", payload, None
    try:
        body = response.json()
    except ValueError:
        body = {"raw": response.text[:2000]}
    if response.status_code != 200:
        return f"### Endpoint error (HTTP {response.status_code})", payload, body
    if "error" in body:
        return f"### The model rejected the input\n{body['error']}", payload, body

    price_brl = body["freight_value_brl"][0]
    route = "within " + seller_state if seller_state == customer_state else f"{seller_state} → {customer_state}"
    summary = (f"## R$ {price_brl:,.2f}\n"
               f"Predicted shipping cost for a {weight_g:,.0f} g item, {route}, about {distance_km:,.0f} km.\n\n"
               f"*Answered in {latency_ms:,.0f} ms by model `{body.get('model', '?')}`.*")
    return summary, payload, body


# -----------------------------------------------------------------------------------------
# User interface
# -----------------------------------------------------------------------------------------
def build_ui():
    connected = bool(ENV_URL and ENV_KEY)
    with gr.Blocks(title="Olist freight-cost tester") as demo:
        gr.Markdown(
            "# Olist shipping-cost predictor\n"
            "Describe an order item and ask **your** Azure ML endpoint what the shipping will cost."
        )
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    seller_state = gr.Dropdown(STATES, value="SP", label="Seller state")
                    customer_state = gr.Dropdown(STATES, value="RJ", label="Customer state")
                    distance_km = gr.Number(value=estimate_distance("SP", "RJ"), minimum=0,
                                            label="Distance (km)",
                                            info="Filled in from the states; change it if you know better.")
                category = gr.Dropdown(CATEGORIES, value="furniture_decor", label="Product category")
                with gr.Row():
                    price = gr.Number(value=249.0, minimum=0, label="Price (R$)")
                    weight_g = gr.Number(value=9500, minimum=0, label="Weight (g)")
                    items_in_order = gr.Number(value=1, minimum=1, precision=0, label="Items in the parcel")
                with gr.Row():
                    length_cm = gr.Number(value=60, minimum=0, label="Length (cm)")
                    height_cm = gr.Number(value=40, minimum=0, label="Height (cm)")
                    width_cm = gr.Number(value=40, minimum=0, label="Width (cm)")
                button = gr.Button("Predict shipping cost", variant="primary")
            with gr.Column(scale=2):
                result = gr.Markdown("Press **Predict shipping cost** to call your endpoint.")
                with gr.Accordion("Request and response (JSON)", open=False):
                    request_json = gr.JSON(label="Request sent")
                    response_json = gr.JSON(label="Response received")

        with gr.Accordion("Connection", open=not connected):
            gr.Markdown(
                "Loaded from `app/.env`. Leave these empty to use it." if connected else
                "**No `.env` found.** Run `scripts/create-app-env.ps1`, or paste your endpoint's "
                "scoring URL and key here (kept in memory only)."
            )
            url_input = gr.Textbox(label="Scoring URL", placeholder="https://<endpoint>.<region>.inference.ml.azure.com/score")
            key_input = gr.Textbox(label="Endpoint key", type="password", placeholder="paste your key")

        inputs = [seller_state, customer_state, category, price, weight_g, length_cm, height_cm, width_cm,
                  distance_km, items_in_order, url_input, key_input]
        button.click(predict, inputs=inputs, outputs=[result, request_json, response_json])
        seller_state.change(estimate_distance, [seller_state, customer_state], distance_km)
        customer_state.change(estimate_distance, [seller_state, customer_state], distance_km)

        gr.Examples(
            label="Example orders (click one, then press Predict)",
            examples=[
                ["SP", "SP", "health_beauty", 59.9, 800, 30, 10, 20, 25, 1],
                ["SP", "RJ", "furniture_decor", 249.0, 9500, 60, 40, 40, 360, 1],
                ["SP", "AM", "computers_accessories", 120.0, 1500, 35, 15, 25, 2700, 1],
            ],
            inputs=[seller_state, customer_state, category, price, weight_g, length_cm, height_cm, width_cm,
                    distance_km, items_in_order],
        )
    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch()
