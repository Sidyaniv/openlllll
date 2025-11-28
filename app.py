import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

# Конфигурация
API_BASE_URL = "http://localhost:8000"
st.set_page_config(page_title="DeltaPos - Рекомендательная система ПСБ")
st.title("ПСБ Банк - Рекомендательная система")

# Инициализация состояния
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

def train_model(dataset_name: str):
    """Отправка запроса на обучение модели"""
    try:
        response = requests.post(f"{API_BASE_URL}/train", json={"dataset_name": dataset_name})
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_predictions(dataset_name: str, customer_ids: list[str] = None):
    """Получение предсказаний от API"""
    try:
        payload = {"dataset_name": dataset_name}
        if customer_ids:
            payload["customer_ids"] = customer_ids
            
        response = requests.post(f"{API_BASE_URL}/predict", json=payload)
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_metrics(dataset_name: str):
    """Получение метрик модели"""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics/{dataset_name}")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Навигация
page = st.sidebar.radio("📊 Navigation", [
    "Model Training", 
    "Product Recommendations", 
    "Model Analytics",
    "How to use"
])

# Страница 1: Обучение модели
if page == "Model Training":
    st.header("🎯 Model Training")
    
    # Получаем доступные датасеты
    datasets = ["retail", "marketplace"]
        # Selectbox для выбора датасета
    st.subheader("1. Select Dataset")
        
    selected_dataset = st.selectbox(
            "Choose a dataset to train the model:",
        options=datasets,
        index=0,
        placeholder="Select dataset...",
        key="dataset_selector"
        )
        
        # Сохраняем выбранный датасет в session state
    if selected_dataset and selected_dataset != st.session_state.selected_dataset:
        st.session_state.selected_dataset = selected_dataset
        st.session_state.model_trained = False
        st.session_state.predictions = None
        
        # Показываем выбранный датасет
    if st.session_state.selected_dataset:
        st.success(f"✅ Selected: {st.session_state.selected_dataset}")
            
            # Информация о датасете
        with st.expander("📊 Dataset Info"):
            st.info(f"**Dataset:** {st.session_state.selected_dataset}")
            st.info(f"**File:** data/{st.session_state.selected_dataset}.pq")
            
            # Кнопка обучения
        st.subheader("2. Train Model")
            
        col1, col2 = st.columns([1, 3])
        with col1:
            train_btn = st.button("🚀 Train Recommendation Model", type="primary", use_container_width=True)
            
        if train_btn:
            with st.spinner("Training model... This may take a few minutes"):
                result = train_model(st.session_state.selected_dataset)
                    
                if result.get('status') == 'success':
                    st.session_state.model_trained = True
                    st.success("🎉 Model trained successfully!")
                        
                        # Показываем базовую информацию
                    st.subheader("📈 Training Results")
                        
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Dataset Size", f"{result.get('dataset_size', 0):,}")
                    with col2:
                        st.metric("Accuracy", f"{result.get('accuracy', 0):.2%}")
                    with col3:
                        st.metric("Features Used", len(result.get('features_used', [])))
                    with col4:
                        st.metric("Status", "Trained ✅")
                        
                        # Показываем использованные фичи
                    with st.expander("🔍 View Features Used"):
                        features = result.get('features_used', [])
                        st.write(f"**Total features:** {len(features)}")
                        for i, feature in enumerate(features, 1):
                            st.write(f"{i}. {feature}")
                                
                else:
                    st.error(f"❌ Training failed: {result.get('message')}")

# Страница 2: Рекомендации продуктов
elif page == "Product Recommendations":
    st.header("🎯 Product Recommendations")
    
    if not st.session_state.selected_dataset:
        st.warning("⚠️ Please select a dataset on the Training page first")
    elif not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first on the Training page")
    else:
        st.success(f"Using model: {st.session_state.selected_dataset}")
        
        # Опции для предсказаний
        st.subheader("1. Prediction Options")
        
        prediction_option = st.radio(
            "Choose prediction scope:",
            ["All Customers", "Specific Customers"],
            horizontal=True
        )
        
        customer_ids_input = None
        if prediction_option == "Specific Customers":
            customer_ids_input = st.text_input(
                "Enter Customer IDs (comma-separated):",
                placeholder="CUST001, CUST002, CUST003"
            )
            if customer_ids_input:
                customer_ids = [cid.strip() for cid in customer_ids_input.split(",")]
                st.info(f"Will predict for {len(customer_ids)} customers")
        
        # Получение предсказаний
        st.subheader("2. Generate Recommendations")
        
        if st.button("🎯 Generate Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                customer_ids_list = customer_ids if prediction_option == "Specific Customers" and customer_ids_input else None
                predictions = get_predictions(st.session_state.selected_dataset, customer_ids_list)
                
                if predictions.get('status') == 'success':
                    st.session_state.predictions = predictions
                    prediction_count = len(predictions['predictions'])
                    st.success(f"✅ Generated recommendations for {prediction_count} customers")
                else:
                    st.error(f"❌ Prediction failed: {predictions.get('message')}")
        
        # Отображение рекомендаций
        if st.session_state.predictions:
            predictions_data = st.session_state.predictions['predictions']
            
            st.subheader("3. Explore Recommendations")
            
            # Выбор клиента для детального просмотра
            customer_options = {p['customer_id']: p for p in predictions_data}
            selected_customer = st.selectbox(
                "🔍 Select Customer for Detailed View:",
                options=list(customer_options.keys()),
                index=0
            )
            
            if selected_customer:
                customer_data = customer_options[selected_customer]
                
                # Профиль клиента
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("👤 Customer Profile")
                    profile_df = pd.DataFrame.from_dict(customer_data['profile'], orient='index', columns=['Value'])
                    st.dataframe(profile_df, use_container_width=True)
                
                with col2:
                    st.subheader("🎯 Top Product Recommendations")
                    rec_df = pd.DataFrame(customer_data['recommendations'])
                    rec_df['rank'] = range(1, len(rec_df) + 1)
                    
                    # Визуализация рекомендаций
                    fig = px.bar(rec_df.head(10), x='probability', y='product', 
                                orientation='h', color='confidence',
                                color_discrete_map={
                                    'high': '#2E8B57', 
                                    'medium': '#FFA500', 
                                    'low': '#DC143C'
                                },
                                title=f"Top Recommendations for {selected_customer}",
                                labels={'probability': 'Probability', 'product': 'Product'})
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Детальная таблица рекомендаций
                st.subheader("📋 Detailed Recommendations")
                
                # Форматируем таблицу для лучшего отображения
                display_df = rec_df.copy()
                display_df['probability'] = display_df['probability'].apply(lambda x: f"{x:.2%}")
                display_df['rank'] = display_df['rank'].astype(int)
                
                st.dataframe(
                    display_df[['rank', 'product', 'probability', 'confidence']],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Кнопка скачивания всех рекомендаций
                st.subheader("4. Export Results")
                
                if st.button("📥 Download All Recommendations as CSV"):
                    all_recs = []
                    for customer in predictions_data:
                        for rec in customer['recommendations']:
                            all_recs.append({
                                'customer_id': customer['customer_id'],
                                'product': rec['product'],
                                'probability': rec['probability'],
                                'confidence': rec['confidence']
                            })
                    
                    export_df = pd.DataFrame(all_recs)
                    csv = export_df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"recommendations_{st.session_state.selected_dataset}.csv",
                        mime="text/csv"
                    )

# Страница 3: Аналитика модели
elif page == "Model Analytics":
    st.header("📊 Model Analytics")
    
    if not st.session_state.selected_dataset:
        st.warning("⚠️ Please select a dataset on the Training page first")
    elif not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first on the Training page")
    else:
        st.success(f"Analyzing: {st.session_state.selected_dataset}")
        
        # Кнопка обновления метрик
        if st.button("🔄 Refresh Analytics", type="primary"):
            with st.spinner("Loading analytics..."):
                metrics = get_metrics(st.session_state.selected_dataset)
                
                if 'error' not in metrics:
                    # Основные метрики
                    st.subheader("📈 Key Metrics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "Model Accuracy", 
                            f"{metrics.get('accuracy', 0):.2%}",
                            delta=None
                        )
                    with col2:
                        st.metric("Dataset Size", f"{metrics.get('dataset_size', 0):,}")
                    with col3:
                        st.metric("Number of Features", len(metrics.get('feature_importance', {})))
                    with col4:
                        st.metric("Model Type", metrics.get('model_type', 'RandomForest'))
                    
                    # Feature Importance
                    st.subheader("🔥 Feature Importance")
                    feature_imp = metrics.get('feature_importance', {})
                    if feature_imp:
                        imp_df = pd.DataFrame.from_dict(feature_imp, orient='index', columns=['importance'])
                        imp_df = imp_df.sort_values('importance', ascending=True)  # Для горизонтального bar chart
                        
                        fig = px.bar(
                            imp_df, 
                            x='importance', 
                            y=imp_df.index,
                            orientation='h',
                            title="Feature Importance Scores",
                            labels={'importance': 'Importance', 'index': 'Features'}
                        )
                        fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Распределение продуктов
                    st.subheader("📦 Product Distribution")
                    product_dist = metrics.get('product_distribution', {})
                    if product_dist:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_pie = px.pie(
                                values=list(product_dist.values()), 
                                names=list(product_dist.keys()),
                                title="Preferred Product Distribution"
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            fig_bar = px.bar(
                                x=list(product_dist.keys()), 
                                y=list(product_dist.values()),
                                title="Product Counts",
                                labels={'x': 'Product', 'y': 'Count'}
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Сегменты клиентов
                    st.subheader("👥 Customer Segments")
                    segment_dist = metrics.get('segment_distribution', {})
                    if segment_dist:
                        # Создаем понятные названия для сегментов
                        segment_names = {
                            0: "Budget-Conscious 🟡",
                            1: "Premium Loyalists 🔵", 
                            2: "Balanced Seekers 🟢",
                            3: "New Explorers 🟠"
                        }
                        
                        segment_data = {
                            segment_names.get(k, f"Segment {k}"): v 
                            for k, v in segment_dist.items()
                        }
                        
                        fig = px.bar(
                            x=list(segment_data.keys()), 
                            y=list(segment_data.values()),
                            title="Customer Segment Distribution",
                            labels={'x': 'Customer Segment', 'y': 'Number of Customers'},
                            color=list(segment_data.keys())
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                else:
                    st.error(f"❌ Failed to load metrics: {metrics.get('error')}")
        else:
            st.info("Click 'Refresh Analytics' to view model metrics and performance")
elif page == 'How to use':
    st.header("🚀 Quick Start Guide:")
    st.subheader("1. Training Page")
    st.info("""
    → Select dataset  
    → Click 'Train Model'
""")
    st.subheader("2. Recommendations Page")
    st.info(""" 
    → Generate predictions  
    → Explore customer recommendations
    """)
    st.subheader("3. Analytics Page")
    st.info(""" 
    → View model metrics  
    → Analyze feature importance
    """)




# Показываем текущий статус в боковой панели
st.sidebar.markdown("---")
st.sidebar.subheader("Current Status")

if st.session_state.selected_dataset:
    st.sidebar.success(f"**Dataset:** {st.session_state.selected_dataset}")
else:
    st.sidebar.warning("**Dataset:** Not selected")

if st.session_state.model_trained:
    st.sidebar.success("**Model:** Trained ✅")
else:
    st.sidebar.warning("**Model:** Not trained")

if st.session_state.predictions:
    pred_count = len(st.session_state.predictions.get('predictions', []))
    st.sidebar.info(f"**Predictions:** {pred_count} customers")
else:
    st.sidebar.info("**Predictions:** None")