# patchcore-MVTech-anomaly-detection

- オリジナルの patchcore のコードから必要な箇所をもってきて、自身が使いやすくしてみました。
- wideresnet だけでなく、mobilenet の中間層を使って軽量化もできるように
- mobilenet の場合、チャネル数少なくなるので、他の箇所のパラメタを変える必要あり
- 当時は、tflite 8ビット軽量化も実施した（pytorch --> tensorflow変換も必要）
