# 1. load data
time_series = load_sensor_timeseries(plot_id)
images = load_images(plot_id)
static = load_static_metadata(plot_id)
yield_target = load_yield(plot_id)

# 2. feature engineering
ts_features = compute_rolling_features(time_series, windows=[3,7,14])
gdd = compute_gdd(time_series['air_temp'])
image_emb = cnn_model.extract_embeddings(images) # per date
merged = align_and_merge(ts_features, image_emb, static)

# 3. train/test split (by season)
train_X, val_X, train_y, val_y = time_split(merged, yield_target)

# 4. model
model = TemporalFusionTransformer(...) # or XGBoost on aggregated features
model.fit(train_X, train_y, validation_data=(val_X, val_y))

# 5. evaluate
preds = model.predict(val_X)
rmse = compute_rmse(val_y, preds)