# Energy-Demand-Forecasting-System-Germany-2027
ML pipeline forecasting Germany's national electricity demand for 2027. Trained on 5 years of real grid data (ENTSO-E), engineered with energy-specific features, and validated against national benchmarks. Built in Python using Random Forest regression.

This project builds an end-to-end machine learning pipeline to forecast Germany's national electricity consumption for 2027.

The model was trained on five years of real historical demand data (2015–2019), using pre-pandemic data deliberately to avoid COVID-era distortions. Data was sourced from ENTSO-E (European electricity grid operator), historical German weather records, and calendar-based behavioral patterns.

Rather than feeding raw data into the model, domain-specific features were engineered — Heating and Cooling Degree Days, day-of-week indicators, and 7-day rolling demand averages — the variables energy analysts actually use in practice.

A Random Forest regression model was trained and evaluated on a train/test split, achieving a Mean Absolute Error of ~56,000 MWh and RMSE of ~85,000 MWh, equivalent to roughly 2% error against Germany's average daily demand of 1.3–1.5 TWh.

Recursive forecasting was then used to simulate daily electricity demand across the full 2027 calendar year, producing a total predicted national demand of ≈ 440.79 TWh.

The most surprising finding was that day-of-week behavioral cycles ranked as a stronger demand predictor than temperature — suggesting human routine imposes the structural load on the grid, while weather modifies around the edges.

Tools: Python · Scikit-learn · Pandas · Matplotlib
