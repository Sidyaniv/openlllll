import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Настройка страницы
st.set_page_config(
    page_title="DeltaPos - Рекомендательная система ПСБ",
    page_icon="🏦",
    layout="wide"
)

# Стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-card {
        background: white;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown('<div class="main-header">🏦 DeltaPos - Рекомендательная система ПСБ</div>', unsafe_allow_html=True)

# Сайдбар для навигации
st.sidebar.title("Навигация")
page = st.sidebar.radio("Выберите раздел:", [
    "📊 Обзор системы",
    "👤 Персональные рекомендации", 
    "📈 Аналитика эффективности",
    "🎯 A/B Тестирование",
    "🚀 Бизнес-impact"
])

# Генерация демо-данных
def generate_demo_data():
    users = [
        {'id': 123456, 'age': 35, 'city': 'Москва', 'segment': 'Premium', 'activity': 'Высокая'},
        {'id': 789012, 'age': 28, 'city': 'Санкт-Петербург', 'segment': 'Массовый', 'activity': 'Средняя'},
        {'id': 345678, 'age': 45, 'city': 'Новосибирск', 'segment': 'Премиум', 'activity': 'Низкая'},
    ]
    
    products = [
        {'name': 'Премиальная карта PSB Premium', 'category': 'cards', 'conv_rate': 0.23},
        {'name': 'Инвестиционный портфель "Сбалансированный"', 'category': 'investments', 'conv_rate': 0.18},
        {'name': 'Накопительный счет "Рост"', 'category': 'savings', 'conv_rate': 0.15},
        {'name': 'Ипотека "Семейная"', 'category': 'mortgage', 'conv_rate': 0.12},
        {'name': 'Страхование путешественников', 'category': 'insurance', 'conv_rate': 0.25},
    ]
    
    return users, products

users, products = generate_demo_data()

if page == "📊 Обзор системы":
    st.header("📊 Обзор рекомендательной системы")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Обработано пользователей", "44M", "135B транзакций")
    with col2:
        st.metric("Точность модели (Precision@5)", "0.38", "+0.12")
    with col3:
        st.metric("Ожидаемая конверсия", "23%", "+8%")
    with col4:
        st.metric("Охват продуктов", "87%", "15%")
    
    # Визуализация данных
    col1, col2 = st.columns(2)
    
    with col1:
        # График распределения пользователей по сегментам
        segments_data = pd.DataFrame({
            'Сегмент': ['Премиум', 'Массовый', 'Спящие', 'Новые'],
            'Количество': [8500000, 28500000, 6500000, 500000]
        })
        fig = px.pie(segments_data, values='Количество', names='Сегмент', 
                     title='Распределение пользователей по сегментам')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # График эффективности по продуктам
        perf_data = pd.DataFrame({
            'Продукт': [p['name'] for p in products],
            'Конверсия': [p['conv_rate'] for p in products]
        })
        fig = px.bar(perf_data, x='Конверсия', y='Продукт', orientation='h',
                     title='Ожидаемая конверсия по продуктам')
        st.plotly_chart(fig, use_container_width=True)

elif page == "👤 Персональные рекомендации":
    st.header("👤 Персональные рекомендации")
    
    # Поиск пользователя
    col1, col2 = st.columns([1, 2])
    
    with col1:
        user_id = st.selectbox("Выберите пользователя:", 
                              [f"{u['id']} ({u['city']}, {u['segment']})" for u in users])
        selected_user = users[[u['id'] for u in users].index(int(user_id.split(' ')[0]))]
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>👤 Профиль пользователя #{selected_user['id']}</h4>
            <p>📍 {selected_user['city']} | 🎯 {selected_user['segment']} | 📊 Активность: {selected_user['activity']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Финансовый профиль
    st.subheader("📈 Финансовый профиль")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Средние расходы", "85,000 руб/мес")
    with col2:
        st.metric("Основные категории", "Рестораны, Путешествия")
    with col3:
        st.metric("Количество транзакций", "28/мес")
    with col4:
        st.metric("Средний чек", "3,040 руб")
    
    # Рекомендации
    st.subheader("🎯 Рекомендуемые продукты")
    
    recommendations = [
        {
            'product': 'Премиальная карта PSB Premium',
            'score': 0.89,
            'reasons': [
                'Подходит под ваш уровень расходов',
                'Повышенный кешбэк в ваших категориях',
                'Бесплатные страховки для путешественников'
            ],
            'features': '5% кешбэк в ресторанах, lounge-доступ'
        },
        {
            'product': 'Инвестиционный портфель "Сбалансированный"',
            'score': 0.76,
            'reasons': [
                'На основе вашей склонности к умеренному риску',
                'Автоматическая ребалансировка',
                'Диверсификация по отраслям'
            ],
            'features': 'Доходность: 12-15% годовых'
        },
        {
            'product': 'Страхование путешественников',
            'score': 0.71,
            'reasons': [
                'Частые поездки в вашей транзакционной истории',
                'Зарубежные онлайн-покупки',
                'Высокая активность в travel-категориях'
            ],
            'features': 'Покрытие 50,000 USD, медицинская помощь'
        }
    ]
    
    for rec in recommendations:
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{rec['product']}**")
                st.markdown(f"*{rec['features']}*")
                
                for reason in rec['reasons']:
                    st.markdown(f"✓ {reason}")
            
            with col2:
                st.metric("Релевантность", f"{rec['score']*100:.0f}%")
                if st.button("Подробнее", key=rec['product']):
                    st.success(f"Открыта детальная информация по {rec['product']}")

elif page == "📈 Аналитика эффективности":
    st.header("📈 Аналитика эффективности")
    
    tab1, tab2, tab3 = st.tabs(["📊 Метрики качества", "👥 Сегментация", "📅 Временные ряды"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Precision-Recall кривая
            metrics_data = pd.DataFrame({
                'Threshold': np.linspace(0.1, 0.9, 9),
                'Precision': [0.25, 0.32, 0.38, 0.41, 0.43, 0.45, 0.44, 0.42, 0.39],
                'Recall': [0.85, 0.78, 0.72, 0.65, 0.58, 0.49, 0.41, 0.33, 0.25]
            })
            fig = px.line(metrics_data, x='Recall', y='Precision', 
                         title='Precision-Recall Curve', markers=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Сравнение алгоритмов
            algo_data = pd.DataFrame({
                'Algorithm': ['LightFM', 'CatBoost', 'ALS', 'Random Forest'],
                'NDCG@10': [0.45, 0.42, 0.38, 0.31],
                'Precision@5': [0.38, 0.35, 0.32, 0.28]
            })
            fig = px.bar(algo_data, x='Algorithm', y=['NDCG@10', 'Precision@5'],
                        title='Сравнение алгоритмов', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Эффективность по сегментам
        segment_perf = pd.DataFrame({
            'Сегмент': ['Премиум', 'Массовый', 'Спящие', 'Новые'],
            'Конверсия': [0.31, 0.22, 0.15, 0.18],
            'Средний чек': [12500, 4500, 2800, 5200],
            'LTV': [185000, 65000, 32000, 48000]
        })
        
        fig = px.scatter(segment_perf, x='Конверсия', y='LTV', size='Средний чек',
                        color='Сегмент', hover_name='Сегмент',
                        title='Эффективность по сегментам клиентов')
        st.plotly_chart(fig, use_container_width=True)

elif page == "🎯 A/B Тестирование":
    st.header("🎯 A/B Тестирование")
    
    st.info("""
    **Текущий эксперимент:** Сравнение новой рекомендательной системы DeltaPos 
    с текущей базовой системой банка
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Группа A (Базовая система)")
        st.metric("Конверсия", "15%", "-")
        st.metric("Средний чек", "8,400 руб", "-")
        st.metric("Кликов на рекомендацию", "12%", "-")
    
    with col2:
        st.subheader("Группа B (DeltaPos)")
        st.metric("Конверсия", "23%", "+8%")
        st.metric("Средний чек", "11,200 руб", "+2,800 руб")
        st.metric("Кликов на рекомендацию", "28%", "+16%")
    
    # Статистическая значимость
    st.subheader("📊 Статистическая значимость")
    
    significance_data = pd.DataFrame({
        'День': range(1, 31),
        'Группа A': np.random.normal(0.15, 0.02, 30),
        'Группа B': np.random.normal(0.23, 0.025, 30)
    })
    
    fig = px.line(significance_data, x='День', y=['Группа A', 'Группа B'],
                  title='Динамика конверсии по дням эксперимента')
    st.plotly_chart(fig, use_container_width=True)

elif page == "🚀 Бизнес-impact":
    st.header("🚀 Бизнес-влияние")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Финансовый эффект")
        
        impact_data = pd.DataFrame({
            'Метрика': ['Доп. выручка', 'Рост LTV', 'Снижение затрат', 'Удержание клиентов'],
            'Значение': [125, 18, 15, 12],
            'Единица': ['млн руб/год', '%', '%', '%']
        })
        
        for _, row in impact_data.iterrows():
            st.metric(f"{row['Метрика']}", f"{row['Значение']} {row['Единица']}")
    
    with col2:
        st.subheader("📈 ROI расчет")
        
        roi_data = pd.DataFrame({
            'Период': ['Месяц 1', 'Месяц 3', 'Месяц 6', 'Год 1'],
            'Инвестиции': [2.5, 5.0, 7.5, 12.0],
            'Доход': [8.3, 25.1, 52.8, 125.4],
            'ROI': [232, 402, 604, 945]
        })
        
        fig = px.line(roi_data, x='Период', y='ROI', markers=True,
                     title='ROI проекта (%)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Влияние на клиентский опыт
    st.subheader("🎯 Влияние на клиентский опыт")
    
    exp_metrics = pd.DataFrame({
        'Метрика': ['NPS', 'CSI', 'Удовлетворенность', 'Лояльность'],
        'Текущее': [45, 68, 72, 65],
        'Целевое': [60, 82, 85, 78],
        'Изменение': [15, 14, 13, 13]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Текущее', x=exp_metrics['Метрика'], y=exp_metrics['Текущее']))
    fig.add_trace(go.Bar(name='Целевое', x=exp_metrics['Метрика'], y=exp_metrics['Целевое']))
    fig.update_layout(title='Метрики клиентского опыта')
    st.plotly_chart(fig, use_container_width=True)

# Футер
st.markdown("---")
st.markdown("**DeltaPos** • Рекомендательная система для банка ПСБ • Хакатон 2024")