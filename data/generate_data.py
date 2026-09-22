"""
Generate a realistic multi-table e-commerce dataset (Olist-like schema).

Planted signals the agent should be able to discover:
  * Q3-2025 revenue dip, driven by electronics cancellations in the South region
  * Freight costs inflating ~35% in the same quarter
  * Review scores collapsing, caused by a spike in late deliveries
"""
import numpy as np, pandas as pd, os
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)
OUT = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(OUT, exist_ok=True)

START = datetime(2024, 1, 1)
DAYS = 730
N_CUSTOMERS, N_SELLERS, N_PRODUCTS, N_ORDERS = 4000, 180, 600, 30000

REGIONS = ["North", "South", "East", "West", "Central"]
REGION_W = [0.18, 0.30, 0.22, 0.17, 0.13]
CATEGORIES = ["electronics", "home_decor", "fashion", "beauty", "sports",
              "books", "toys", "grocery", "furniture"]
CAT_W = [0.22, 0.14, 0.16, 0.10, 0.09, 0.07, 0.08, 0.08, 0.06]
CITIES = {
    "North": ["Delhi", "Chandigarh", "Lucknow"],
    "South": ["Bengaluru", "Chennai", "Hyderabad"],
    "East": ["Kolkata", "Bhubaneswar", "Guwahati"],
    "West": ["Mumbai", "Pune", "Ahmedabad"],
    "Central": ["Bhopal", "Nagpur", "Raipur"],
}
CHANNELS = ["organic", "paid_search", "social", "email", "affiliate"]
CHANNEL_W = [0.34, 0.26, 0.18, 0.14, 0.08]


def ids(prefix, n):
    return [f"{prefix}_{i:05d}" for i in range(n)]


def make_customers():
    region = RNG.choice(REGIONS, N_CUSTOMERS, p=REGION_W)
    return pd.DataFrame({
        "customer_id": ids("cust", N_CUSTOMERS),
        "customer_city": [RNG.choice(CITIES[r]) for r in region],
        "customer_region": region,
        "signup_date": [START + timedelta(days=int(RNG.integers(0, DAYS - 30)))
                        for _ in range(N_CUSTOMERS)],
        "acquisition_channel": RNG.choice(CHANNELS, N_CUSTOMERS, p=CHANNEL_W),
    })


def make_sellers():
    region = RNG.choice(REGIONS, N_SELLERS, p=REGION_W)
    return pd.DataFrame({
        "seller_id": ids("sell", N_SELLERS),
        "seller_city": [RNG.choice(CITIES[r]) for r in region],
        "seller_region": region,
        "seller_tier": RNG.choice(["gold", "silver", "bronze"], N_SELLERS,
                                  p=[0.2, 0.45, 0.35]),
        "onboarded_date": [START + timedelta(days=int(RNG.integers(-400, DAYS - 60)))
                           for _ in range(N_SELLERS)],
    })


def make_products(sellers):
    cat = RNG.choice(CATEGORIES, N_PRODUCTS, p=CAT_W)
    base = {"electronics": 6500, "home_decor": 1400, "fashion": 1100, "beauty": 700,
            "sports": 1800, "books": 400, "toys": 900, "grocery": 350, "furniture": 9000}
    price = [max(99, RNG.normal(base[c], base[c] * 0.35)) for c in cat]
    weight = [max(50, RNG.normal({"furniture": 12000, "electronics": 1800}.get(c, 600), 300))
              for c in cat]
    return pd.DataFrame({
        "product_id": ids("prod", N_PRODUCTS),
        "product_category": cat,
        "list_price": np.round(price, 2),
        "product_weight_g": np.round(weight, 0),
        "seller_id": RNG.choice(sellers.seller_id, N_PRODUCTS),
    })


def make_orders(customers):
    day_idx = np.arange(DAYS)
    dates = [START + timedelta(days=int(d)) for d in day_idx]
    trend = 1.0 + 0.00055 * day_idx
    seasonal = 1 + 0.22 * np.sin(2 * np.pi * (day_idx % 365) / 365 - 1.1)
    festive = np.where(((day_idx % 365) > 285) & ((day_idx % 365) < 320), 1.45, 1.0)
    w = trend * seasonal * festive
    order_days = RNG.choice(day_idx, N_ORDERS, p=w / w.sum())

    cust = RNG.choice(customers.customer_id, N_ORDERS)
    cmap = customers.set_index("customer_id")["customer_region"].to_dict()
    region = np.array([cmap[c] for c in cust])

    purchase = pd.to_datetime([dates[d] for d in order_days])
    q3 = (order_days >= 547) & (order_days <= 638)          # Q3 2025
    primary_cat = RNG.choice(CATEGORIES, N_ORDERS, p=CAT_W)

    # PLANTED SIGNAL 1: South + electronics collapse in Q3 2025
    drop_mask = (q3 & (region == "South") & (primary_cat == "electronics")
                 & (RNG.random(N_ORDERS) < 0.62))

    status = RNG.choice(["delivered", "shipped", "canceled"], N_ORDERS,
                        p=[0.93, 0.045, 0.025])
    status = np.where(drop_mask, "canceled", status)

    ship = RNG.gamma(3.0, 2.2, N_ORDERS) + 1
    # PLANTED SIGNAL 2: logistics degradation in Q3 => late deliveries
    ship = ship + np.where(q3, RNG.gamma(2.0, 2.0, N_ORDERS), 0)

    approved = purchase + pd.to_timedelta(RNG.integers(1, 30, N_ORDERS), unit="h")
    delivered = purchase + pd.to_timedelta(np.round(ship, 2), unit="D")
    estimated = purchase + pd.to_timedelta(RNG.integers(5, 16, N_ORDERS), unit="D")

    orders = pd.DataFrame({
        "order_id": ids("ord", N_ORDERS),
        "customer_id": cust,
        "order_status": status,
        "order_purchase_ts": purchase,
        "order_approved_ts": approved,
        "order_delivered_ts": pd.Series(delivered).where(
            pd.Series(status) == "delivered"),
        "order_estimated_delivery_ts": estimated,
    })
    orders["_primary_cat"] = primary_cat
    orders["_q3"] = q3
    return orders


def make_items(orders, products):
    rows = []
    pidx = products.set_index("product_id")
    n_items = RNG.choice([1, 1, 1, 2, 2, 3], len(orders))
    prod_pool = products.product_id.to_numpy()
    by_cat = {c: products[products.product_category == c].product_id.to_numpy()
              for c in products.product_category.unique()}

    for oid, n, cat, q3 in zip(orders.order_id, n_items,
                               orders._primary_cat, orders._q3):
        for k in range(n):
            pool = by_cat.get(cat, prod_pool) if k == 0 else prod_pool
            pid = RNG.choice(pool if len(pool) else prod_pool)
            row = pidx.loc[pid]
            price = float(row.list_price) * float(RNG.normal(1.0, 0.07))
            # PLANTED SIGNAL 3: freight inflation of ~35% in Q3
            freight = (60 + float(row.product_weight_g) * 0.09) * (1.35 if q3 else 1.0)
            rows.append((oid, k + 1, pid, row.seller_id,
                         round(max(49, price), 2), round(freight, 2)))
    return pd.DataFrame(rows, columns=["order_id", "order_item_id", "product_id",
                                       "seller_id", "price", "freight_value"])


def make_payments(items):
    tot = items.groupby("order_id")[["price", "freight_value"]].sum().sum(axis=1)
    types = RNG.choice(["upi", "credit_card", "debit_card", "wallet", "cod"],
                       len(tot), p=[0.38, 0.28, 0.12, 0.10, 0.12])
    return pd.DataFrame({
        "order_id": tot.index,
        "payment_type": types,
        "payment_installments": np.where(types == "credit_card",
                                         RNG.integers(1, 7, len(tot)), 1),
        "payment_value": np.round(tot.to_numpy(), 2),
    })


def make_reviews(orders):
    d = orders[orders.order_status == "delivered"].copy()
    late = pd.to_datetime(d.order_delivered_ts) > pd.to_datetime(d.order_estimated_delivery_ts)
    base = RNG.normal(4.5, 0.8, len(d))
    score = np.clip(np.round(base - np.where(late, RNG.normal(1.9, 0.6, len(d)), 0)), 1, 5)
    keep = RNG.random(len(d)) < 0.72
    d = d[keep]
    return pd.DataFrame({
        "review_id": ids("rev", int(keep.sum())),
        "order_id": d.order_id.to_numpy(),
        "review_score": score[keep].astype(int),
        "review_creation_ts": pd.to_datetime(d.order_delivered_ts)
                              + pd.to_timedelta(RNG.integers(1, 10, len(d)), unit="D"),
    })


def main():
    customers = make_customers()
    sellers = make_sellers()
    products = make_products(sellers)
    orders = make_orders(customers)
    items = make_items(orders, products)
    payments = make_payments(items)
    reviews = make_reviews(orders)
    orders = orders.drop(columns=["_primary_cat", "_q3"])

    for name, df in [("customers", customers), ("sellers", sellers),
                     ("products", products), ("orders", orders),
                     ("order_items", items), ("payments", payments),
                     ("reviews", reviews)]:
        path = os.path.join(OUT, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"{name:12s} {len(df):>7,} rows -> {path}")


if __name__ == "__main__":
    main()