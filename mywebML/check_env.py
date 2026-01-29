try:
    import pandas
    import sklearn
    import joblib
    print("Environment OK")
except ImportError as e:
    print(f"ImportError: {e}")
