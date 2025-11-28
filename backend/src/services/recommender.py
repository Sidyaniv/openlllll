import pandas

class RecommenderService:
    def __init__(self, model):
        self.model = model

    def recommend(self, req):
        retail = pandas.read_parquet(req.retail_events_path)
        market = pandas.read_parquet(req.marketplace_events_path)
        payments = pandas.read_parquet(req.payment_events_path)
        receipts = pandas.read_parquet(req.receipts_path)
        retail_items = pandas.read_parquet(req.retail_items_path)
        marketplace_items = pandas.read_parquet(req.marketplace_items_path)

        # Инференс модели
        recommendations = self.model.predict(
            retail=retail,
            market=market,
            payments=payments,
            receipts=receipts,
            retail_items=retail_items,
            marketplace_items=marketplace_items,
        )

        return recommendations
