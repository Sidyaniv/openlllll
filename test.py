import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import random
from datetime import datetime

# Конфигурация
st.set_page_config(page_title="DeltaPos - Рекомендательная система ПСБ", layout="wide")
st.title("ПСБ Банк - Рекомендательная система")

# Инициализация session state
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'mock_mode' not in st.session_state:
    st.session_state.mock_mode = True

# Мок-функции
def mock_train_model(dataset_name: str):
    """Мок-функция для обучения модели"""
    st.toast("🔧 Используются тестовые данные...")
    
    # Имитация задержки
    import time
    time.sleep(2)
    
    return {
        "status": "success",
        "dataset_size": random.randint(10000, 50000),
        "accuracy": round(random.uniform(0.75, 0.95), 4),
        "features_used": [
            "customer_age", "income_level", "transaction_frequency",
            "avg_transaction_amount", "product_preferences",
            "geographic_region", "credit_score", "loyalty_duration"
        ],
        "model_type": "RandomForest",
        "training_time": f"{random.randint(30, 120)} seconds"
    }

def mock_get_predictions(dataset_name: str, customer_ids: list[str] = None):
    """Мок-функция для получения предсказаний"""
    st.toast("🔧 Генерация тестовых рекомендаций...")
    
    products = ["Кредитная карта Premium", "Ипотека", "Автокредит", 
                "Инвестиционный счет", "Страхование", "Сберегательный счет",
                "Дебетовая карта Gold", "Потребительский кредит", "Бизнес-счет"]
    
    if customer_ids is None:
        customer_ids = [f"КЛИЕНТ{str(i).zfill(3)}" for i in range(1, 21)]
    
    predictions = []
    for customer_id in customer_ids[:10]:
        customer_predictions = []
        for product in random.sample(products, 5):
            prob = round(random.uniform(0.1, 0.99), 4)
            confidence = "высокая" if prob > 0.7 else "средняя" if prob > 0.4 else "низкая"
            customer_predictions.append({
                "product": product,
                "probability": prob,
                "confidence": confidence
            })
        
        customer_predictions.sort(key=lambda x: x["probability"], reverse=True)
        
        predictions.append({
            "customer_id": customer_id,
            "profile": {
                "возраст": random.randint(25, 65),
                "доход": f"{random.randint(30000, 150000)} руб.",
                "регион": random.choice(["Москва", "СПб", "Новосибирск", "Екатеринбург"]),
                "клиент_с": f"{random.randint(1, 10)} лет",
                "сегмент": random.choice(["Премиум", "Стандарт", "Новый"]),
                "продуктов": random.randint(1, 5)
            },
            "recommendations": customer_predictions[:5]
        })
    
    return {
        "status": "success",
        "predictions": predictions
    }

def mock_get_metrics(dataset_name: str):
    """Мок-функция для получения метрик"""
    feature_importance = {
        "частота_транзакций": round(random.uniform(0.1, 0.3), 4),
        "уровень_дохода": round(random.uniform(0.08, 0.25), 4),
        "возраст": round(random.uniform(0.05, 0.15), 4),
        "кредитный_рейтинг": round(random.uniform(0.07, 0.2), 4),
        "регион": round(random.uniform(0.03, 0.1), 4)
    }
    
    product_distribution = {
        "Кредитные карты": random.randint(1000, 5000),
        "Ипотека": random.randint(500, 2000),
        "Автокредиты": random.randint(800, 3000),
        "Сберегательные счета": random.randint(2000, 6000)
    }
    
    return {
        "accuracy": round(random.uniform(0.82, 0.94), 4),
        "dataset_size": random.randint(25000, 75000),
        "model_type": "RandomForest",
        "feature_importance": feature_importance,
        "product_distribution": product_distribution
    }

# Основные функции
def train_model(dataset_name: str):
    if st.session_state.mock_mode:
        return mock_train_model(dataset_name)
    else:
        try:
            response = requests.post(f"http://localhost:8000/train", json={"dataset_name": dataset_name})
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

def get_predictions(dataset_name: str, customer_ids: list[str] = None):
    if st.session_state.mock_mode:
        return mock_get_predictions(dataset_name, customer_ids)
    else:
        try:
            payload = {"dataset_name": dataset_name}
            if customer_ids:
                payload["customer_ids"] = customer_ids
            response = requests.post(f"http://localhost:8000/predict", json=payload)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

def get_metrics(dataset_name: str):
    if st.session_state.mock_mode:
        return mock_get_metrics(dataset_name)
    else:
        try:
            response = requests.get(f"http://localhost:8000/metrics/{dataset_name}")
            return response.json()
        except Exception as e:
            return {"error": str(e)}

# Боковая панель
st.sidebar.title("Навигация")
page = st.sidebar.radio("Выберите страницу:", [
    "Обучение модели", 
    "Рекомендации", 
    "Аналитика",
    "Инструкция"
])

st.sidebar.markdown("---")
st.sidebar.subheader("Настройки")
st.session_state.mock_mode = st.sidebar.checkbox("Тестовый режим", value=True)

# Главная логика приложения
if page == "Обучение модели":
    st.header("🎯 Обучение модели")
    
    # Выбор датасета
    dataset = st.selectbox(
        "Выберите датасет для обучения:",
        ["розничные_клиенты", "корпоративные_клиенты", "интернет_банк"],
        index=0
    )
    
    st.session_state.selected_dataset = dataset
    st.success(f"✅ Выбран датасет: {dataset}")
    
    # Кнопка обучения - ВСЕГДА видна
    st.subheader("Запуск обучения")
    if st.button("🚀 Обучить модель", type="primary", use_container_width=True):
        with st.spinner("Идет обучение модели... Это может занять несколько минут"):
            result = train_model(dataset)
            
            if result.get('status') == 'success':
                st.session_state.model_trained = True
                st.balloons()
                st.success("🎉 Модель успешно обучена!")
                
                # Показываем результаты
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Размер данных", f"{result.get('dataset_size', 0):,}")
                with col2:
                    st.metric("Точность", f"{result.get('accuracy', 0):.2%}")
                with col3:
                    st.metric("Кол-во фич", len(result.get('features_used', [])))
                with col4:
                    st.metric("Статус", "Обучена ✅")
                
            else:
                st.error(f"❌ Ошибка обучения: {result.get('message')}")

elif page == "Рекомендации":
    st.header("🎯 Рекомендации продуктов")
    
    if not st.session_state.selected_dataset:
        st.warning("⚠️ Сначала выберите датасет на странице 'Обучение модели'")
        st.info("Перейдите на страницу 'Обучение модели', выберите датасет и нажмите 'Обучить модель'")
    else:
        st.success(f"Используется модель: {st.session_state.selected_dataset}")
        
        # Опции предсказания
        st.subheader("Настройки рекомендаций")
        prediction_option = st.radio(
            "Для кого генерировать рекомендации:",
            ["Все клиенты", "Конкретные клиенты"],
            horizontal=True
        )
        
        customer_ids = None
        if prediction_option == "Конкретные клиенты":
            ids_input = st.text_input(
                "ID клиентов (через запятую):",
                placeholder="КЛИЕНТ001, КЛИЕНТ002, КЛИЕНТ003"
            )
            if ids_input:
                customer_ids = [cid.strip() for cid in ids_input.split(",")]
                st.info(f"Будет обработано {len(customer_ids)} клиентов")
        
        # Кнопка генерации - ВСЕГДА видна при выбранном датасете
        st.subheader("Генерация рекомендаций")
        if st.button("🎯 Сгенерировать рекомендации", type="primary"):
            with st.spinner("Генерация рекомендаций..."):
                predictions = get_predictions(st.session_state.selected_dataset, customer_ids)
                
                if predictions.get('status') == 'success':
                    st.session_state.predictions = predictions
                    pred_count = len(predictions['predictions'])
                    st.success(f"✅ Сгенерировано рекомендаций для {pred_count} клиентов")
                    
                    # Сразу показываем результаты
                    st.subheader("Результаты рекомендаций")
                    
                    # Выбор клиента для детализации
                    pred_data = predictions['predictions']
                    customer_options = [p['customer_id'] for p in pred_data]
                    selected_customer = st.selectbox("Выберите клиента:", customer_options)
                    
                    if selected_customer:
                        customer_data = next(p for p in pred_data if p['customer_id'] == selected_customer)
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.subheader("👤 Профиль клиента")
                            for key, value in customer_data['profile'].items():
                                st.write(f"**{key}:** {value}")
                        
                        with col2:
                            st.subheader("🎯 Топ рекомендаций")
                            rec_df = pd.DataFrame(customer_data['recommendations'])
                            
                            fig = px.bar(rec_df, x='probability', y='product', 
                                        orientation='h',
                                        title=f"Топ рекомендаций для {selected_customer}",
                                        color='confidence',
                                        color_discrete_map={
                                            'высокая': '#2E8B57', 
                                            'средняя': '#FFA500', 
                                            'низкая': '#DC143C'
                                        })
                            fig.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error(f"❌ Ошибка генерации: {predictions.get('message')}")

elif page == "Аналитика":
    st.header("📊 Аналитика модели")
    
    if not st.session_state.selected_dataset:
        st.warning("⚠️ Сначала выберите датасет на странице 'Обучение модели'")
    elif not st.session_state.model_trained:
        st.warning("⚠️ Сначала обучите модель на странице 'Обучение модели'")
    else:
        st.success(f"Анализ модели: {st.session_state.selected_dataset}")
        
        # Кнопка загрузки метрик - ВСЕГДА видна
        if st.button("🔄 Загрузить аналитику", type="primary"):
            with st.spinner("Загрузка метрик..."):
                metrics = get_metrics(st.session_state.selected_dataset)
                
                if 'error' not in metrics:
                    # Основные метрики
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Точность модели", f"{metrics.get('accuracy', 0):.2%}")
                    with col2:
                        st.metric("Размер данных", f"{metrics.get('dataset_size', 0):,}")
                    with col3:
                        st.metric("Тип модели", metrics.get('model_type', 'RandomForest'))
                    with col4:
                        st.metric("Кол-во фич", len(metrics.get('feature_importance', {})))
                    
                    # Важность фич
                    st.subheader("🔥 Важность признаков")
                    feature_imp = metrics.get('feature_importance', {})
                    if feature_imp:
                        imp_df = pd.DataFrame(list(feature_imp.items()), columns=['Feature', 'Importance'])
                        imp_df = imp_df.sort_values('Importance', ascending=True)
                        
                        fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                                    title="Важность признаков в модели")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Распределение продуктов
                    st.subheader("📦 Распределение продуктов")
                    product_dist = metrics.get('product_distribution', {})
                    if product_dist:
                        fig = px.pie(values=list(product_dist.values()), names=list(product_dist.keys()),
                                    title="Распределение предпочтений продуктов")
                        st.plotly_chart(fig, use_container_width=True)
                        
                else:
                    st.error(f"❌ Ошибка загрузки: {metrics.get('error')}")

elif page == "Инструкция":
    st.header("📖 Инструкция по использованию")
    
    st.info("""
    ### 🚀 Быстрый старт:
    
    1. **Страница 'Обучение модели'**
       - Выберите датасет
       - Нажмите "Обучить модель"
       - Дождитесь завершения обучения
    
    2. **Страница 'Рекомендации'**
       - Выберите scope (все клиенты или конкретные)
       - Нажмите "Сгенерировать рекомендации"
       - Просмотрите результаты
    
    3. **Страница 'Аналитика'**
       - Нажмите "Загрузить аналитику"
       - Изучите метрики модели
    """)
    
    st.warning("""
    💡 **Примечание:** Сейчас включен тестовый режим. 
    Все данные генерируются случайным образом для демонстрации.
    """)

# Статус в боковой панели
st.sidebar.markdown("---")
st.sidebar.subheader("Текущий статус")

status_dataset = st.session_state.selected_dataset or "Не выбран"
status_model = "Обучена ✅" if st.session_state.model_trained else "Не обучена"
status_predictions = "Есть" if st.session_state.predictions else "Нет"

st.sidebar.info(f"""
**Датасет:** {status_dataset}
**Модель:** {status_model}
**Рекомендации:** {status_predictions}
""")

# Дебаг информация
if st.sidebar.checkbox("Показать отладочную информацию"):
    st.sidebar.write("Session State:")
    st.sidebar.json({
        "selected_dataset": st.session_state.selected_dataset,
        "model_trained": st.session_state.model_trained,
        "predictions": bool(st.session_state.predictions),
        "mock_mode": st.session_state.mock_mode
    })