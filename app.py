import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

# Конфигурация
API_BASE_URL = "http://localhost:8000"
st.set_page_config(
    page_title="DeltaPos - Рекомендательная система ПСБ", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомные стили для красивого UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .recommendation-card {
        background: white;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown("""
<div class="main-header">
    <h1>ПСБ Банк - Рекомендательная система DeltaPos</h1>
    <p style="margin-top: 0.5rem; font-size: 1.1rem; opacity: 0.9;">Персонализированные рекомендации банковских продуктов</p>
</div>
""", unsafe_allow_html=True)

# Инициализация состояния
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'selected_test_dataset' not in st.session_state:
    st.session_state.selected_test_dataset = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'training_result' not in st.session_state:
    st.session_state.training_result = None

def check_model_status():
    """Проверка статуса модели на бэкенде"""
    try:
        # Пытаемся получить метрики - если модель обучена, метрики будут доступны
        response = requests.get(f"{API_BASE_URL}/metrics/train", timeout=2)
        if response.status_code == 200:
            data = response.json()
            # Проверяем, что метрики действительно есть (не ошибка)
            if data.get("status") != 'error':
                return True
        return False
    except requests.exceptions.RequestException:
        return False
    except Exception:
        return False

def train_model():
    """Отправка запроса на обучение модели (используется train датасет)"""
    try:
        response = requests.post(f"{API_BASE_URL}/train", json={"dataset_name": "train"})
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_predictions(test_dataset_name: str, customer_ids: Optional[list[str]] = None):
    """Получение предсказаний от API для тестового датасета"""
    try:
        payload = {"dataset_name": test_dataset_name}
        if customer_ids:
            payload["customer_ids"] = customer_ids
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=30)
        response.raise_for_status()  # Вызовет исключение для статусов 4xx, 5xx
        return response.json()
    except requests.exceptions.HTTPError as e:
        # Обработка HTTP ошибок (400, 404, 500, etc.)
        try:
            error_data = e.response.json()
            detail = error_data.get("detail", str(e))
            return {"status": "error", "message": detail}
        except:
            return {"status": "error", "message": f"HTTP {e.response.status_code}: {str(e)}"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Ошибка соединения: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"Неизвестная ошибка: {str(e)}"}

def get_metrics(dataset_name: str):
    """Получение метрик модели"""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics/{dataset_name}", timeout=5)
        response.raise_for_status()  # Вызовет исключение для статусов 4xx, 5xx
        data = response.json()
        # Убираем error из ответа если он None (Pydantic может добавить его)
        if data.get('error') is None:
            data.pop('error', None)
        return data
    except requests.exceptions.HTTPError as e:
        # Обработка HTTP ошибок (404, 500, etc.)
        try:
            error_data = e.response.json()
            return {"error": error_data.get("detail", str(e))}
        except:
            return {"error": f"HTTP {e.response.status_code}: {str(e)}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Ошибка соединения: {str(e)}"}
    except Exception as e:
        return {"error": f"Неизвестная ошибка: {str(e)}"}

# Навигация
st.sidebar.subheader("Навигация")
page = st.sidebar.radio("", [
    "Обучение модели", 
    "Рекомендации", 
    "Аналитика модели"
])

# Страница 1: Обучение модели
if page == "Обучение модели":
    st.header("Обучение модели")
    
    # Проверяем статус модели при загрузке страницы
    if 'model_status_checked' not in st.session_state:
        st.session_state.model_trained = check_model_status()
        st.session_state.model_status_checked = True
    
    # Показываем статус модели
    if st.session_state.model_trained:
        st.success("Модель уже обучена!")
        
        # Загружаем метрики с бэкенда для отображения
        with st.spinner("Загрузка информации о модели..."):
            metrics = get_metrics("train")
        
        # Используем training_result если есть, иначе метрики с бэкенда
        display_data = st.session_state.training_result if st.session_state.training_result else metrics
        
        # Показываем результаты обучения только если есть данные
        if display_data and not display_data.get('error'):
            st.subheader("Результаты обучения")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Размер датасета", f"{display_data.get('dataset_size', 0):,}")
            with col2:
                st.metric("Точность", f"{display_data.get('accuracy', 0):.2%}")
            with col3:
                features_used = display_data.get('features_used', [])
                if not features_used and display_data.get('feature_importance'):
                    # Если нет features_used, используем ключи из feature_importance
                    features_used = list(display_data.get('feature_importance', {}).keys())
                st.metric("Кол-во признаков", len(features_used))
            with col4:
                st.metric("Тип модели", display_data.get('model_type', 'Unknown'))
                
            # Показываем использованные фичи
            if features_used:
                with st.expander("Просмотр использованных признаков"):
                    st.write(f"**Всего признаков:** {len(features_used)}")
                    for i, feature in enumerate(features_used, 1):
                        st.write(f"{i}. {feature}")
            
            # Показываем важность признаков если доступна
            if display_data.get('feature_importance'):
                with st.expander("Важность признаков"):
                    feature_importance = display_data.get('feature_importance', {})
                    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"**{feature}:** {importance:.4f}")
        else:
            st.warning("Не удалось загрузить информацию о модели. Попробуйте переобучить модель.")
        
        st.markdown("---")
        st.info("Вы можете перейти к генерации рекомендаций или переобучить модель")
        
        # Кнопка переобучения
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            retrain_btn = st.button("Переобучить модель", type="secondary", width='stretch')
        
        if retrain_btn:
            with st.spinner("Переобучение модели... Это может занять несколько минут"):
                result = train_model()
                
                if result.get('status') == 'success':
                    st.session_state.model_trained = True
                    st.session_state.training_result = result
                    st.success("Модель успешно переобучена!")
                    st.rerun()
                else:
                    st.error(f"Ошибка обучения: {result.get('message')}")
    else:
        # Модель не обучена - показываем кнопку обучения
        st.info("Модель еще не обучена. Нажмите кнопку ниже для начала обучения.")
        
        st.markdown("---")
        st.subheader("Запуск обучения")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            train_btn = st.button("Обучить модель", type="primary", width='stretch')
        
        if train_btn:
            with st.spinner("Обучение модели... Это может занять несколько минут"):
                result = train_model()
                
                if result.get('status') == 'success':
                    st.session_state.model_trained = True
                    st.session_state.training_result = result
                    st.success("Модель успешно обучена!")
                    
                    # Показываем результаты
                    st.subheader("Результаты обучения")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Размер датасета", f"{result.get('dataset_size', 0):,}")
                    with col2:
                        st.metric("Точность", f"{result.get('accuracy', 0):.2%}")
                    with col3:
                        st.metric("Кол-во признаков", len(result.get('features_used', [])))
                    with col4:
                        st.metric("Тип модели", result.get('model_type', 'Unknown'))
                    
                    # Показываем использованные фичи
                    with st.expander("Просмотр использованных признаков"):
                        features = result.get('features_used', [])
                        st.write(f"**Всего признаков:** {len(features)}")
                        for i, feature in enumerate(features, 1):
                            st.write(f"{i}. {feature}")
                else:
                    st.error(f"Ошибка обучения: {result.get('message')}")

# Страница 2: Рекомендации
elif page == "Рекомендации":
    st.header("Генерация рекомендаций")
    
    # Проверяем, обучена ли модель
    if not st.session_state.model_trained:
        # Проверяем на бэкенде
        st.session_state.model_trained = check_model_status()
    
    if not st.session_state.model_trained:
        st.warning("Модель не обучена. Пожалуйста, сначала обучите модель на странице 'Обучение модели'.")
        if st.button("Перейти к обучению"):
            st.session_state.model_status_checked = False
            st.rerun()
    else:
        st.success("Модель обучена и готова к использованию")
        
        # Список тестовых датасетов
        test_datasets = {
            "test": "Тестовый датасет (Test Dataset)",
            "test_1": "Тестовый датасет 1",
            "test_2": "Тестовый датасет 2",
            "test_3": "Тестовый датасет 3"
        }
        
        st.subheader("1. Выбор тестового датасета")
        
        selected_test_dataset = st.selectbox(
            label="Выберите тестовый датасет для генерации рекомендаций:",
            options=list(test_datasets.keys()),
            format_func=lambda x: test_datasets[x],
            index=0,
            key="test_dataset_selector",
            help="Выберите датасет с клиентами, для которых нужно сгенерировать рекомендации"
        )
        
        st.session_state.selected_test_dataset = selected_test_dataset
        
        if selected_test_dataset:
            st.markdown(f"""
            <div class="info-box">
                <strong>Выбран датасет:</strong> {test_datasets[selected_test_dataset]}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("2. Настройка генерации рекомендаций")
            
            # Выбор режима: все пользователи или конкретные
            generation_mode = st.radio(
                label="Режим генерации:",
                options=["Все пользователи из датасета", "Конкретные пользователи"],
                horizontal=True,
                key="generation_mode",
                help="Выберите, для кого генерировать рекомендации"
            )
            
            customer_ids = None
            customer_ids_input = None
            
            if generation_mode == "Конкретные пользователи":
                customer_ids_input = st.text_area(
                    label="Введите ID пользователей (по одному на строку или через запятую):",
                    placeholder="CUST001\nCUST002\nCUST003\n\nили\n\nCUST001, CUST002, CUST003",
                    key="customer_ids_input",
                    help="Введите ID пользователей, для которых нужно сгенерировать рекомендации",
                    height=100
                )
                
                if customer_ids_input:
                    # Обработка разных форматов ввода
                    lines = customer_ids_input.replace(',', '\n').split('\n')
                    customer_ids = [cid.strip() for cid in lines if cid.strip()]
                    customer_ids = [cid for cid in customer_ids if cid]  # Убираем пустые
                    
                    if customer_ids:
                        st.info(f"Будет обработано {len(customer_ids)} пользователей: {', '.join(customer_ids[:5])}{'...' if len(customer_ids) > 5 else ''}")
                    else:
                        st.warning("Не найдено валидных ID пользователей")
            
            st.markdown("---")
            st.subheader("3. Генерация рекомендаций")
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                generate_btn = st.button("Сгенерировать рекомендации", type="primary", width='stretch')
            
            # Генерация при нажатии кнопки
            if generate_btn:
                if generation_mode == "Конкретные пользователи" and not customer_ids:
                    st.warning("Пожалуйста, введите ID пользователей для генерации рекомендаций")
                else:
                    with st.spinner("Генерация рекомендаций..."):
                        predictions = get_predictions(selected_test_dataset, customer_ids)
                        
                        if predictions.get('status') == 'success':
                            st.session_state.predictions = predictions
                            st.session_state.last_generated_dataset = selected_test_dataset
                            prediction_count = len(predictions.get('predictions', []))
                            st.success(f"Сгенерировано рекомендаций для {prediction_count} клиентов")
                        else:
                            error_msg = predictions.get('message') or "Неизвестная ошибка"
                            st.error(f"Ошибка генерации: {error_msg}")
                            
                            # Дополнительная информация для отладки
                            with st.expander("Детали ошибки"):
                                st.json(predictions)
            
            # Отображение рекомендаций
            if st.session_state.predictions and st.session_state.predictions.get('status') == 'success':
                predictions_data = st.session_state.predictions.get('predictions', [])
                
                if predictions_data:
                    st.markdown("---")
                    st.subheader("4. Просмотр рекомендаций")
                    
                    # Статистика по рекомендациям
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Всего клиентов", len(predictions_data))
                    with col2:
                        total_recs = sum(len(c.get('recommendations', [])) for c in predictions_data)
                        st.metric("Всего рекомендаций", total_recs)
                    with col3:
                        avg_recs = total_recs / len(predictions_data) if predictions_data else 0
                        st.metric("Среднее на клиента", f"{avg_recs:.1f}")
                    
                    # Выбор клиента для детального просмотра
                    customer_options = {p['customer_id']: p for p in predictions_data}
                    
                    # Улучшенный выбор клиента с поиском
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        selected_customer = st.selectbox(
                            label="Выберите клиента для детального просмотра:",
                            options=list(customer_options.keys()),
                            index=0,
                            key="selected_customer_view",
                            help="Выберите клиента из списка для просмотра его рекомендаций"
                        )
                    with col2:
                        # Быстрый поиск по ID
                        search_id = st.text_input(
                            label="Быстрый поиск по ID:",
                            placeholder="CUST001",
                            key="customer_search",
                            help="Введите ID клиента для быстрого поиска"
                        )
                        if search_id:
                            if search_id in customer_options:
                                selected_customer = search_id
                                st.success(f"Найден: {search_id}")
                            else:
                                st.warning(f"Клиент {search_id} не найден")
                    
                    if selected_customer:
                        customer_data = customer_options[selected_customer]
                        
                        # Профиль клиента и рекомендации
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <h3>Профиль клиента</h3>
                                <p><strong>ID:</strong> {selected_customer}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            profile = customer_data.get('profile', {})
                            if profile:
                                # Красивое отображение профиля
                                for key, value in profile.items():
                                    st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                                
                                # Дополнительная информация
                                st.markdown("---")
                                st.markdown("### Статистика")
                                rec_count = len(customer_data.get('recommendations', []))
                                st.metric("Количество рекомендаций", rec_count)
                                
                                if rec_count > 0:
                                    top_rec = customer_data['recommendations'][0]
                                    st.metric("Топ рекомендация", top_rec.get('product', 'N/A'))
                                    st.metric("Вероятность", f"{top_rec.get('probability', 0):.2%}")
                        
                        with col2:
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <h3>Топ рекомендаций</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            rec_df = pd.DataFrame(customer_data.get('recommendations', []))
                            if not rec_df.empty:
                                rec_df['rank'] = range(1, len(rec_df) + 1)
                                
                                # Улучшенная визуализация рекомендаций
                                fig = px.bar(
                                    rec_df.head(10), 
                                    x='probability', 
                                    y='product', 
                                    orientation='h', 
                                    color='confidence',
                                    color_discrete_map={
                                        'high': '#2E8B57', 
                                        'medium': '#FFA500', 
                                        'low': '#DC143C'
                                    },
                                    title=f"Топ рекомендаций для {selected_customer}",
                                    labels={'probability': 'Вероятность', 'product': 'Продукт'},
                                    text='probability'
                                )
                                fig.update_traces(
                                    texttemplate='%{text:.1%}',
                                    textposition='outside',
                                    marker_line_color='white',
                                    marker_line_width=1.5
                                )
                                fig.update_layout(
                                    yaxis={'categoryorder': 'total ascending'},
                                    height=400,
                                    showlegend=True,
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)'
                                )
                                st.plotly_chart(fig, width='stretch')
                        
                        # Детальная таблица рекомендаций
                        st.markdown("---")
                        st.subheader("Детальные рекомендации")
                        if not rec_df.empty:
                            display_df = rec_df.copy()
                            display_df['probability'] = display_df['probability'].apply(lambda x: f"{x:.2%}")
                            display_df['rank'] = display_df['rank'].astype(int)
                            
                            # Улучшенное отображение таблицы
                            st.dataframe(
                                display_df[['rank', 'product', 'probability', 'confidence']],
                                width='stretch',
                                hide_index=True,
                                column_config={
                                    "rank": "Ранг",
                                    "product": "Продукт",
                                    "probability": "Вероятность",
                                    "confidence": st.column_config.TextColumn(
                                        "Уверенность",
                                        help="Уровень уверенности в рекомендации"
                                    )
                                }
                            )
                            
                            # Дополнительная информация
                            col1, col2 = st.columns(2)
                            with col1:
                                high_conf = len(rec_df[rec_df['confidence'] == 'high'])
                                st.metric("Высокая уверенность", high_conf)
                            with col2:
                                avg_prob = rec_df['probability'].mean()
                                st.metric("Средняя вероятность", f"{avg_prob:.2%}")
                    
                    # Экспорт результатов
                    st.markdown("---")
                    st.subheader("5. Экспорт результатов")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Скачать все рекомендации как CSV", width='stretch'):
                            all_recs = []
                            for customer in predictions_data:
                                for rec in customer.get('recommendations', []):
                                    all_recs.append({
                                        'customer_id': customer['customer_id'],
                                        'product': rec['product'],
                                        'probability': rec['probability'],
                                        'confidence': rec['confidence']
                                    })
                            
                            export_df = pd.DataFrame(all_recs)
                            csv = export_df.to_csv(index=False)
                            
                            st.download_button(
                                label="Скачать CSV файл",
                                data=csv,
                                file_name=f"recommendations_{selected_test_dataset}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                width='stretch'
                            )
                    
                    with col2:
                        # Показываем предпросмотр данных для экспорта
                        if st.button("Предпросмотр данных", width='stretch'):
                            preview_data = []
                            for customer in predictions_data[:5]:  # Первые 5 для предпросмотра
                                for rec in customer.get('recommendations', [])[:3]:  # Первые 3 рекомендации
                                    preview_data.append({
                                        'customer_id': customer['customer_id'],
                                        'product': rec['product'],
                                        'probability': f"{rec['probability']:.2%}",
                                        'confidence': rec['confidence']
                                    })
                            
                            if preview_data:
                                preview_df = pd.DataFrame(preview_data)
                                st.dataframe(preview_df, width='stretch')
                                st.caption(f"Показано {len(preview_data)} записей из {sum(len(c.get('recommendations', [])) for c in predictions_data)}")

# Страница 3: Аналитика модели
elif page == "Аналитика модели":
    st.header("Аналитика модели")
    
    # Проверяем, обучена ли модель
    if not st.session_state.model_trained:
        st.session_state.model_trained = check_model_status()
    
    if not st.session_state.model_trained:
        st.warning("Модель не обучена. Пожалуйста, сначала обучите модель.")
    else:
        st.success("Анализ обученной модели")
        
        # Автоматическая загрузка метрик
        with st.spinner("Загрузка метрик модели..."):
            metrics = get_metrics("train")
            
            # Проверяем наличие ошибки (не просто ключ, а реальное значение)
            if not metrics.get('error'):
                # Основные метрики
                st.subheader("Ключевые метрики")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Точность модели", f"{metrics.get('accuracy', 0):.2%}")
                with col2:
                    st.metric("Размер датасета", f"{metrics.get('dataset_size', 0):,}")
                with col3:
                    st.metric("Кол-во признаков", len(metrics.get('feature_importance', {})))
                with col4:
                    st.metric("Тип модели", metrics.get('model_type', 'Unknown'))
                
                # Важность признаков
                st.subheader("Важность признаков")
                feature_imp = metrics.get('feature_importance', {})
                if feature_imp:
                    imp_df = pd.DataFrame.from_dict(feature_imp, orient='index', columns=['importance'])
                    imp_df = imp_df.sort_values('importance', ascending=True)
                    
                    fig = px.bar(
                        imp_df, 
                        x='importance', 
                        y=imp_df.index,
                        orientation='h',
                        title="Важность признаков в модели",
                        labels={'importance': 'Важность', 'index': 'Признаки'}
                    )
                    fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, width='stretch')
                
                # Распределение продуктов
                st.subheader("Распределение продуктов")
                product_dist = metrics.get('product_distribution', {})
                if product_dist:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_pie = px.pie(
                            values=list(product_dist.values()), 
                            names=list(product_dist.keys()),
                            title="Распределение предпочтений продуктов"
                        )
                        st.plotly_chart(fig_pie, width='stretch')
                    
                    with col2:
                        fig_bar = px.bar(
                            x=list(product_dist.keys()), 
                            y=list(product_dist.values()),
                            title="Количество по продуктам",
                            labels={'x': 'Продукт', 'y': 'Количество'}
                        )
                        st.plotly_chart(fig_bar, width='stretch')
                
                # Сегменты клиентов
                st.subheader("Сегменты клиентов")
                segment_dist = metrics.get('segment_distribution', {})
                if segment_dist:
                    segment_names = {
                        0: "Бюджетные 🟡",
                        1: "Премиум 🔵", 
                        2: "Сбалансированные 🟢",
                        3: "Новые 🟠"
                    }
                    
                    segment_data = {
                        segment_names.get(k, f"Сегмент {k}"): v 
                        for k, v in segment_dist.items()
                    }
                    
                    fig = px.bar(
                        x=list(segment_data.keys()), 
                        y=list(segment_data.values()),
                        title="Распределение сегментов клиентов",
                        labels={'x': 'Сегмент', 'y': 'Количество клиентов'},
                        color=list(segment_data.keys())
                    )
                    st.plotly_chart(fig, width='stretch')
            else:
                error_msg = metrics.get('error') or "Неизвестная ошибка"
                st.error(f"Ошибка загрузки метрик: {error_msg}")
                
                # Показываем дополнительную информацию для отладки
                with st.expander("🔍 Детали ошибки"):
                    st.json(metrics)

# Показываем текущий статус в боковой панели
st.sidebar.markdown("---")
st.sidebar.subheader("Текущий статус")

if st.session_state.model_trained:
    st.sidebar.success("**Модель:** Обучена")
else:
    st.sidebar.warning("**Модель:** Не обучена")

if st.session_state.selected_test_dataset:
    st.sidebar.info(f"**Тестовый датасет:** {st.session_state.selected_test_dataset}")

if st.session_state.predictions:
    pred_count = len(st.session_state.predictions.get('predictions', []))
    st.sidebar.info(f"**Рекомендации:** {pred_count} клиентов")
else:
    st.sidebar.info("**Рекомендации:** Нет")
