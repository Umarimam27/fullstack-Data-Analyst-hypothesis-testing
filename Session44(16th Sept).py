# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pandas.plotting import andrews_curves, parallel_coordinates
import warnings

warnings.filterwarnings("ignore")

# ---- App Title ----
st.set_page_config(layout="wide")
st.title("📊 Comprehensive Sales Data Dashboard")
st.markdown("Upload your sales CSV and explore >20 visualizations grouped into tabs.")

# ---- File uploader ----
file = st.file_uploader("Upload Sales CSV", type=["csv"])
if not file:
    st.info("Upload a CSV file to begin. (Use example dataset or your file.)")
    st.stop()

# ---- Load DataFrame ----
try:
    df = pd.read_csv(file)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

# ---- Normalize column names ----
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Detect date-like columns
date_candidates = [c for c in df.columns if "date" in c or "day" in c or "month" in c]
for col in date_candidates:
    try:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    except Exception:
        pass

# Quick preview & dtype summary
with st.expander("Data preview & summary", expanded=False):
    st.dataframe(df.head(100))
    st.write("Columns:", list(df.columns))
    st.write("Dtypes:")
    st.write(df.dtypes)

# Utility: numeric columns
def numeric_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

# ---- Tabs ----
tab_basic, tab_adv, tab_multi = st.tabs(["📊 Basic", "📈 Advanced", "🔬 Multivariate / Special"])

# -------------------------
# BASIC PLOTS
# -------------------------
with tab_basic:
    st.header("Basic Visualizations")

    # 1) Bar: Units Sold per Product
    st.subheader("Bar: Units Sold per Product")
    if {"product_name", "units_sold"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(10, 4))
        df.plot(kind="bar", x="product_name", y="units_sold", ax=ax, legend=False)
        ax.set_xlabel("Product")
        ax.set_ylabel("Units Sold")
        ax.set_title("Units Sold per Product")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Need columns: product_name, units_sold")

    # 2) Line: Units Sold Over Time
    st.subheader("Line: Units Sold Over Time")
    date_col = next((c for c in df.columns if c.endswith("date")), None)
    if date_col and "units_sold" in df.columns:
        tmp = df[[date_col, "units_sold"]].dropna().sort_values(by=date_col)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(tmp[date_col], tmp["units_sold"], marker="o")
        ax.set_xlabel(date_col)
        ax.set_ylabel("Units Sold")
        ax.set_title("Units Sold Over Time")
        fig.autofmt_xdate()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Need a date column and 'units_sold'")

    # 3) Pie: Units Sold by Category
    st.subheader("Pie: Units Sold by Category")
    if {"category", "units_sold"}.issubset(df.columns):
        cat = df.groupby("category")["units_sold"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(cat, labels=cat.index, autopct="%1.1f%%", startangle=90)
        ax.set_title("Units Sold by Category")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Need columns: category, units_sold")

    # 4) Histogram
    st.subheader("Histogram: Units Sold Distribution")
    if "units_sold" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df["units_sold"].dropna(), bins=12)
        ax.set_title("Units Sold Distribution")
        ax.set_xlabel("Units Sold")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Need column: units_sold")

    # 5) Stacked Bar: Product x Category
    st.subheader("Stacked Bar: Units Sold per Product (by category)")
    if {"product_name", "category", "units_sold"}.issubset(df.columns):
        pivot = df.pivot_table(index="product_name", columns="category", values="units_sold", aggfunc="sum").fillna(0)
        fig, ax = plt.subplots(figsize=(12, 5))
        pivot.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Stacked Units Sold by Product and Category")
        ax.set_xlabel("Product")
        ax.set_ylabel("Units Sold")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Category")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Skipping stacked bar — need product_name, category, units_sold")
        
    # 6) Strip Plot
    st.subheader("Strip Plot: Units Sold by Category")
    if {"category", "units_sold"}.issubset(df.columns): 
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.stripplot(x="category", y="units_sold", data=df, jitter=True, ax=ax)
        ax.set_title("Units Sold by Category (Strip)")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Need category & units_sold for strip plot")   
        
    # 7) Joint Plot
st.subheader("Joint Plot: sale_date_num vs units_sold")
if {"sale_date_num", "units_sold"}.issubset(df.columns):
    df_clean = df.dropna(subset=["sale_date_num", "units_sold"])
    j = sns.jointplot(
        x="sale_date_num",
        y="units_sold",
        data=df_clean,
        kind="reg",
        height=6,
        color="teal"
    )
    j.fig.suptitle("Units Sold vs Sale Date (Joint)", y=1.02)
    st.pyplot(j.fig)
    plt.close(j.fig)
else:
    st.info("Need 'sale_date_num' & 'units_sold' for jointplot")

# -------------------------
# ADVANCED PLOTS
# -------------------------
with tab_adv:
    st.header("Advanced Visualizations")

    # 6) Scatter
    st.subheader("Scatter: Units Sold vs Revenue (or other numeric)")
    if "units_sold" in df.columns:
        numeric = [c for c in numeric_cols(df) if c != "units_sold"]
        y_col = "revenue" if "revenue" in df.columns else (numeric[0] if numeric else None)
        if y_col:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(df["units_sold"], df[y_col], alpha=0.6)
            ax.set_xlabel("Units Sold")
            ax.set_ylabel(y_col)
            ax.set_title(f"Units Sold vs {y_col}")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("No numeric column to compare with units_sold")
    else:
        st.warning("Need 'units_sold' column")

    # 7) Hexbin
    st.subheader("Hexbin: Density between two numeric columns")
    if len(numeric_cols(df)) >= 2:
        n1, n2 = numeric_cols(df)[:2]
        fig, ax = plt.subplots(figsize=(8, 5))
        hb = ax.hexbin(df[n1], df[n2], gridsize=30)
        fig.colorbar(hb, ax=ax, label="counts")
        ax.set_xlabel(n1); ax.set_ylabel(n2)
        ax.set_title(f"Hexbin: {n1} vs {n2}")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Need at least 2 numeric columns for hexbin")

    # 8) KDE
    st.subheader("KDE: units_sold distribution")
    if "units_sold" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.kdeplot(df["units_sold"].dropna(), fill=True, ax=ax)
        ax.set_title("Units Sold KDE")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No 'revenue' column for KDE")

    # 9) Box
    st.subheader("Box: units_sold by Category")
if {"category", "units_sold"}.issubset(df.columns):
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(x="category", y="units_sold", data=df, ax=ax)
    ax.set_title("Units Sold by Category (Box)")
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("Need 'category' & 'units_sold' for box plot")

    # 10) Violin
    st.subheader("Violin: Units Sold by Category")
    if {"category", "units_sold"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.violinplot(x="category", y="units_sold", data=df, ax=ax)
        ax.set_title("Units Sold by Category (Violin)")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Need category & units_sold for violin")

    # 11) Jointplot
    st.subheader("Jointplot: Sale_date_num vs units_sold")

if {"Sale_date_num", "units_sold"}.issubset(df.columns):
    # Drop rows with missing values for clean plotting
    df_clean = df.dropna(subset=["Sale_date_num", "units_sold"])

    st.subheader("Jointplot: Sale_date_num vs units_sold")

if {"Sale_date_num", "units_sold"}.issubset(df.columns):
    # Drop rows with missing values for clean plotting
    df_clean = df.dropna(subset=["Sale_date_num", "units_sold"])

    # Create jointplot
    j = sns.jointplot(
        x="Sale_date_num",
        y="units_sold",
        data=df_clean,
        kind="hex",   # 🔹 options: "scatter", "reg", "hex", "kde"
        height=7,
        cmap="mako"   # nice colormap
    )

    j.fig.suptitle("Sale_date_num vs units_sold", y=1.02)

    # Safely render in Streamlit
    fig = j.fig
    st.pyplot(fig)
    plt.close(fig)

else:
    st.info("Need 'Sale_date_num' & 'units_sold' columns for jointplot")
    
    
    # 12) Pairplot
    st.subheader("Pairplot: Numeric Relationships")
    if len(numeric_cols(df)) >= 2:
        pair = sns.pairplot(df[numeric_cols(df)].dropna())
        st.pyplot(pair.fig)
        plt.close(pair.fig)
    else:
        st.info("Need 2+ numeric columns for pairplot")

    # 13) Heatmap
    st.subheader("Heatmap: Correlation Matrix")
    nums = numeric_cols(df)
    if len(nums) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[nums].corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation heatmap")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Not enough numeric columns for heatmap")

    # 14) Stripplot
    st.subheader("Stripplot: Units Sold by Product Line")
    if {"category", "units_sold"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.stripplot(x="category", y="units_sold", data=df, jitter=True, ax=ax)
        ax.set_title("Units Sold by Product Line (Strip)")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Need category & units_sold for stripplot")

# -------------------------
# MULTIVARIATE / SPECIAL
# -------------------------
with tab_multi:
    st.header("Multivariate & Special Visualizations")

    # 15) 3D Scatter
st.subheader("3D Scatter: units_sold vs revenue vs first numeric")
if "units_sold" in df.columns:
    # Pick revenue or any other numeric column
    if "revenue" in df.columns:
        ycol = "revenue"
    else:
        nums = [c for c in numeric_cols(df) if c != "units_sold"]
        ycol = nums[0] if nums else None

    third = next((c for c in numeric_cols(df) if c not in ("units_sold", ycol)), None)

    if ycol and third:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        mask = df[["units_sold", ycol, third]].dropna()
        ax.scatter(mask["units_sold"], mask[ycol], mask[third], alpha=0.6)
        ax.set_xlabel("Units Sold"); ax.set_ylabel(ycol); ax.set_zlabel(third)
        ax.set_title("3D Scatter")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Need at least 3 numeric columns (including units_sold) for 3D scatter")
else:
    st.info("Need 'units_sold' column for 3D scatter")
    
    # 16) Treemap
    st.subheader("Treemap: Units Sold by Category")
    if {"category", "units_sold"}.issubset(df.columns):
        cat = df.groupby("category")["units_sold"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        squarify.plot(sizes=cat["units_sold"], label=cat["category"], alpha=0.7, ax=ax)
        ax.axis("off")
        ax.set_title("Treemap - Units Sold by Category")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Need category & units_sold for treemap")

    # 17) Hexbin (alternate)
    st.subheader("Hexbin (alternate): density plot for two numeric columns")
    if len(numeric_cols(df)) >= 2:
        n1, n2 = numeric_cols(df)[:2]
        fig, ax = plt.subplots(figsize=(8, 5))
        hb = ax.hexbin(df[n1], df[n2], gridsize=25)
        fig.colorbar(hb, ax=ax)
        ax.set_xlabel(n1); ax.set_ylabel(n2)
        ax.set_title("Hexbin plot")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Need 2+ numeric columns for hexbin")

    # 18) Andrews Curves
st.subheader("Andrews Curves (multivariate by category)")
if "category" in df.columns and len(numeric_cols(df)) >= 2:
    try:
        plot_df = df[numeric_cols(df) + ["category"]].dropna()
        fig, ax = plt.subplots(figsize=(10, 6))
        andrews_curves(plot_df, "category", ax=ax)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Andrews curves error: {e}")
else:
    st.info("Need 'category' plus numeric columns for Andrews curves")


    # 19) Parallel Coordinates
    st.subheader("Parallel Coordinates (first numeric columns)")
if "category" in df.columns and len(numeric_cols(df)) >= 3:
    try:
        cols = numeric_cols(df)[:4]
        plot_df = df[cols + ["category"]].dropna()
        fig, ax = plt.subplots(figsize=(12, 5))
        parallel_coordinates(plot_df, "category", ax=ax, colormap=plt.cm.tab10)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Parallel coordinates error: {e}")
else:
    st.info("Need 'category' plus >=3 numeric cols for parallel coordinates")


    # 20) Swarm plot
    st.subheader("Swarm: Units Sold by Category (swarm)")
    if {"category", "units_sold"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.swarmplot(x="category", y="units_sold", data=df, ax=ax)
        ax.set_title("Swarm: Units Sold by Category")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Need category & units_sold for swarmplot")

    # 21) Extras: Correlation & Pivot Download
    st.subheader("Extras: Correlation & Pivot Download")
    if len(numeric_cols(df)) >= 2:
        corr = df[numeric_cols(df)].corr()
        st.write("Correlation matrix (numeric columns):")
        st.dataframe(corr)
    if {"product_name", "category", "units_sold"}.issubset(df.columns):
        pivot = df.pivot_table(index="product_name", columns="category", values="units_sold", aggfunc="sum").fillna(0)
        st.write("Pivot table (units_sold by product_name x category):")
        st.dataframe(pivot)
        csv = pivot.to_csv().encode("utf-8")
        st.download_button("Download pivot CSV", data=csv, file_name="pivot_units_by_product_category.csv", mime="text/csv")
