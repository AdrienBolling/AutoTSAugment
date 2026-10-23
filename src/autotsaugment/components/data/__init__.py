"""Benchmark data (WP1): GIFT-Eval from the Hugging Face Hub as fixed window sets.

Modules: ``download`` (Hub listing, size plan, download), ``arrow`` (Arrow shards to flat
series), ``windows`` (splits, window cutting, padding, naive scores, ``WindowSet`` assembly),
``cache`` (per-dataset window files) and ``gifteval`` (the ``Benchmark`` and its config).
"""
