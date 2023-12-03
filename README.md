MLOps project repository.

Рассмотрена задача классификации на примере датасета Ирисов Фишера (Iris Dataset)
В качестве обучаемой модели выступает простой классификатор с тремя линейными полносвязными слоями


Для использования доступны следующие команды:

python .\commands.py train - обучение и сохранение модели
python .\commands.py infer - infer модели, сохраненной в .pt
python .\commands.py run_server - serving onnx модели
python .\commands.py run_mlflow_tracking_server - запуск локального mlflow tracking сервера


Настраивать параметры обучения и инференса можно через конфиг configs/config.yaml
