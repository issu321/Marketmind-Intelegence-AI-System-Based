# MarketMind Test Datasets

This package contains 8 example CSV files designed specifically to test all features of your MarketMind Competitive Intelligence and Trend Prediction Platform.

---

## 📊 Dataset Overview

| File | Rows | Purpose | Key Columns |
|------|------|---------|-------------|
| `marketmind_sales_forecast.csv` | 730 | **Forecasting Engine** - Time series prediction | date, sales, units_sold, marketing_spend, customer_count |
| `marketmind_competitor_intelligence.csv` | 120 | **Competitor Analysis** - SWOT, HHI, threat scoring | date, competitor_name, market_share, revenue, price, reviews |
| `marketmind_consumer_feedback.csv` | 500 | **Consumer Intelligence** - Sentiment analysis | review_text, rating, sentiment, product_name, category |
| `marketmind_opportunity_detection.csv` | 4,380 | **Opportunity Detection** - Growth pockets, market readiness | date, category, revenue, marketing_roi, market_penetration |
| `marketmind_scenario_simulation.csv` | 365 | **Scenario Simulation** - What-if analysis | date, revenue, units_sold, marketing_spend, conversion_rate |
| `marketmind_executive_dashboard.csv` | 365 | **Executive Command Center** - KPI tracking | date, revenue, gross_profit, net_profit, total_customers, NPS |
| `marketmind_multi_region.csv` | 360 | **Visualization & Segmentation** - Multi-dimensional charts | date, region, segment, revenue, churn_rate, satisfaction_score |
| `marketmind_social_media.csv` | 1,000 | **Brand Monitoring** - Topic extraction, social listening | mention_text, platform, topic, sentiment, likes, shares, reach |

---

## 🎯 How to Test Each Feature

### 1. Forecasting Engine (`marketmind_sales_forecast.csv`)
**Upload to:** Datasets → Upload → Select `marketmind_sales_forecast.csv`
**Test:** Forecasting → Select dataset → Date: `date`, Value: `sales`
**Expected:** Auto-model selection, confidence intervals, trend insights

### 2. Competitor Intelligence (`marketmind_competitor_intelligence.csv`)
**Upload to:** Datasets → Upload
**Test:** Competitors → Select dataset → Competitor: `competitor_name`, Value: `market_share`
**Expected:** SWOT analysis, HHI market concentration, threat scores, radar charts

### 3. Consumer Intelligence (`marketmind_consumer_feedback.csv`)
**Upload to:** Datasets → Upload
**Test:** Consumer → Select dataset → Text: `review_text`, Rating: `rating`
**Expected:** Sentiment distribution (45% positive, 30% neutral, 25% negative), emotion breakdown, topic extraction, brand health score

### 4. Opportunity Detection (`marketmind_opportunity_detection.csv`)
**Upload to:** Datasets → Upload
**Test:** Opportunities → Select dataset → Date: `date`, Value: `revenue`, Category: `category`
**Expected:** Growth pocket detection (Health category shows surge after day 500), opportunity scoring, risk assessment

### 5. Scenario Simulation (`marketmind_scenario_simulation.csv`)
**Upload to:** Datasets → Upload
**Test:** Simulation → Select dataset → Value: `revenue`, Date: `date`
**Try scenarios:** Price Increase, Marketing Increase, Demand Surge, New Product Launch
**Expected:** Impact analysis, comparison charts, revenue projections

### 6. Executive Dashboard (`marketmind_executive_dashboard.csv`)
**Upload to:** Datasets → Upload
**Test:** Executive → View unified KPIs
**Expected:** Business health scores, growth metrics, customer analytics, strategic recommendations

### 7. Visualizations (`marketmind_multi_region.csv`)
**Upload to:** Datasets → Upload
**Test:** Visualizations → Select dataset
**Try charts:** Line (revenue over time), Bar (by region), Scatter (satisfaction vs revenue), Pie (market share), Box (churn by segment)
**Expected:** Interactive Plotly charts with dark theme

### 8. Social Media / Brand Monitoring (`marketmind_social_media.csv`)
**Upload to:** Datasets → Upload
**Test:** Consumer → Select dataset → Text: `mention_text`
**Expected:** Topic extraction, platform-wise sentiment, influence scoring

---

## 🔍 Data Quality Features Built-In

Each dataset includes realistic data characteristics for testing:
- **Trends:** Upward/downward growth patterns
- **Seasonality:** Weekly, monthly, yearly cycles
- **Anomalies:** Deliberate outliers and spikes
- **Missing patterns:** Natural data gaps
- **Categorical diversity:** Multiple segments, regions, products
- **Text variety:** Realistic reviews with varying sentiment

---

## 📈 Sample Statistics

| Dataset | Date Range | Records | Unique Values |
|---------|-----------|---------|---------------|
| Sales Forecast | 2023-01-01 to 2024-12-30 | 730 | 5 columns |
| Competitor Intel | 2023-01-01 to 2024-12-01 | 120 | 5 competitors |
| Consumer Feedback | 2023-01-01 to 2024-12-31 | 500 | 5 products, 5 categories |
| Opportunities | 2023-01-01 to 2024-12-30 | 4,380 | 6 categories |
| Scenario Sim | 2023-01-01 to 2023-12-31 | 365 | 10 metrics |
| Executive | 2023-01-01 to 2023-12-31 | 365 | 16 KPIs |
| Multi-Region | 2023-01-01 to 2024-12-01 | 360 | 5 regions × 3 segments |
| Social Media | 2023-01-01 to 2024-12-31 | 1,000 | 6 platforms, 10 topics |

---

## 💡 Pro Tips for Testing

1. **Upload multiple datasets** to test the dashboard stats (Datasets count, Forecasts count, etc.)
2. **Run forecasts on different horizons** (30, 90, 180, 365 days)
3. **Try all chart types** in the Visualization center
4. **Generate reports** after running analyses
5. **Test admin features** with the default admin account
6. **Check correlation heatmaps** using datasets with multiple numeric columns

---

Generated for MarketMind v1.0 Testing
