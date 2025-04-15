# Computer Vision Project: Animal-10 Classification

Проект з класифікації зображень тварин за допомогою нейронних мереж. Метою є побудова моделі, яка здатна класифікувати зображення тварин на основі датасету "Animals-10", що містить зображення 10 різних видів тварин. Таких як:
```text
data classes: ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo'], number of classes: 10
data classes: ['dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel'], number of classes: 10
```

## Опис проекту
Цей проект включає наступні етапи:
1. Підготовка середовища для тренування моделі.
2. Завантаження і обробка датасету "Animals-10".
3. Проведення аналізу даних, включаючи аугментацію та візуалізацію.
4. Розробка архітектури нейронної мережі для класифікації зображень.
5. Тренування моделі та оцінка її ефективності.

## Опис датасету "Animals-10"
- **Кількість зображень**: 26179.
- **Кількість класів**: 10.
- **Якість зображень**: середня.
- **Кількість зображень по класах**: від 1446 до 4863 зображень.
```python
{
     'cane': 4863,
     'ragno': 4821,
     'gallina': 3098,
     'cavallo': 2623,
     'farfalla': 2112,
     'mucca': 1866,
     'scoiattolo': 1862,
     'pecora': 1820,
     'gatto': 1668,
     'elefante': 1446
 }
```
![Number of images per classes bar plot](imgs/classes_distr.png)
Цей датасет можна використовувати для тестування моделей розпізнавання зображень, таких як домашні CNN (які можуть досягати точності близько 80%) або більш складні моделі, як Google Inception (яка досягла 98% точності). Датасет також може бути корисним для симуляції розумної галереї для дослідників, таких як біологи, для автоматичного розпізнавання видів тварин.
### Структура каталогу:
```text
Animals-10/raw-img/
│
├── cane/
│   ├── image_001.jpg
│   ├── ...
│   └── image_n.jpg
│
├── cavallo/
│   ├── image_001.jpg
│   ├── ...
│   └── image_n.jpg
│
├── elefante/
│   ├── image_001.jpg
│   ├── ...
│   └── image_n.jpg
│
├── farfalla/
│   ├── image_001.jpg
│   ├── ...
│   └── image_n.jpg
│
├── gallina/
│   ├── image_001.jpg
│   ├── ...
│   └── image_n.jpg
│
├── gatto/
│   ├── image_001.jpg
│   ├── ...
│   └── image_n.jpg
│
├── mucca/
│   ├── image_001.jpg
│   ├── ...
│   └── image_n.jpg
│
├── pecora/
│   ├── image_001.jpg
│   ├── ...
│   └── image_n.jpg
│
├── ragno/
│   ├── image_001.jpg
│   ├── ...
│   └── image_n.jpg
│
└── scoiattolo/
│   ├── image_001.jpg
│   ├── ...
│   └── image_n.jpg
```

### Формати файлів
![Number of images per extension bar plot](imgs/extensions_distr.png)


## Аналіз якості даних:
```text
number of corrupted images: 0
number of duplicates: 0
```

## Візуалізація даних:
![Some images from dataset](imgs/some_images.png)
![Some images from dataset](imgs/some_images_by_classes.png)

## Результати тренування простої моделі:
### Модель:
```python
class CNNModel(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, padding=1, filter_size=3, pool_size=2):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=filter_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=filter_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=filter_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=filter_size, padding=padding)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
```
### Результати трейну:
![img.png](imgs/img.png)

### Результати валідації:
![img.png](imgs/valid_hw2.png)

### Візуалізація кількох передбачень:
![img.png](imgs/res_vis_hw2.png)

### Кількість правильно помічених даних тесту:
![img.png](imgs/test_result_hw2.png)

### Матриця плутанини:
![img.png](imgs/conf_matrix_hw2.png)

## Результати тренування моделі з вручну підібраними параметрами:
### Модель:
```python
class CNNModel(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, filters=[32, 64, 128, 256], fc_units=[512, 256, 128]):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(filters[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(filters[2], filters[3], kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(filters[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self._to_linear = None
        self.convs(torch.zeros(1, in_channels, 64, 64))

        self.fc1 = nn.Linear(self._to_linear, fc_units[0])
        self.fc2 = nn.Linear(fc_units[0], fc_units[1])
        self.fc3 = nn.Linear(fc_units[1], fc_units[2])
        self.fc4 = nn.Linear(fc_units[2], num_classes)

    def convs(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = self.pool4(x)

        if self._to_linear is None:
            self._to_linear = x.view(x.size(0), -1).size(1)
        return x

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
```
### Результати трейну:
![img.png](imgs/train_hw3_manual.png)

### Результати валідації:
![img.png](imgs/val_hw3_manual.png)

### Візуалізація кількох передбачень:
![img.png](imgs/images_hw3_manual.png)

### Кількість правильно помічених даних тесту:
```python
              precision    recall  f1-score   support

         dog       0.69      0.75      0.72       952
       horse       0.73      0.73      0.73       529
    elephant       0.64      0.72      0.68       297
   butterfly       0.78      0.83      0.80       442
     chicken       0.80      0.85      0.83       603
         cat       0.68      0.47      0.55       348
         cow       0.66      0.68      0.67       376
       sheep       0.72      0.53      0.61       366
      spider       0.87      0.86      0.86       973
    squirrel       0.63      0.66      0.64       349

    accuracy                           0.74      5235
   macro avg       0.72      0.71      0.71      5235
weighted avg       0.74      0.74      0.74      5235
```

### Матриця плутанини:
![img.png](imgs/conf_matrix_hw3_manual.png)

## Результати тренування моделі з автоматично підібраними параметрами:
### Модель:
```python
class CNNModel(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, filters=[32, 64, 128, 256], fc_units=[512, 256, 128]):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(filters[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(filters[2], filters[3], kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(filters[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self._to_linear = None
        self.convs(torch.zeros(1, in_channels, 64, 64))

        self.fc1 = nn.Linear(self._to_linear, fc_units[0])
        self.fc2 = nn.Linear(fc_units[0], fc_units[1])
        self.fc3 = nn.Linear(fc_units[1], fc_units[2])
        self.fc4 = nn.Linear(fc_units[2], num_classes)

    def convs(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = self.pool4(x)

        if self._to_linear is None:
            self._to_linear = x.view(x.size(0), -1).size(1)
        return x

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
```

### Параметри:
```python
{
    'lr': 0.0003359902842986991,
     'batch_size': 128,
     'conv1_filters': 16,
     'conv2_filters': 64,
     'conv3_filters': 256,
     'conv4_filters': 512,
     'fc1_units': 768,
     'fc2_units': 384,
     'fc3_units': 128,
     'optimizer': 'Adam'
 }
```
### Результати трейну:
![img.png](imgs/train_hw3_auto.png)

### Результати валідації:
![img.png](imgs/val_hw3_auto.png)

### Візуалізація кількох передбачень:
![img.png](imgs/images_hw3_auto.png)

### Кількість правильно помічених даних тесту:
```python
              precision    recall  f1-score   support

dog           0.70         0.79    0.74       951
horse         0.75         0.78    0.77       520
elephant      0.78         0.73    0.76       317
butterfly     0.82         0.81    0.82       443
chicken       0.87         0.81    0.84       623
cat           0.66         0.55    0.60       330
cow           0.69         0.65    0.67       387
sheep         0.67         0.68    0.68       339
spider        0.86         0.88    0.87       949
squirrel      0.72         0.68    0.70       376

accuracy                           0.77       5235
macro avg     0.75         0.74    0.74       5235
weighted avg  0.77         0.77    0.76       5235
```

### Матриця плутанини:
![img.png](imgs/conf_matrix_hw3_auto.png)

![img.png](imgs/dependance_hw3.png)

## Результати тренування претренованих моделей:
### Модель:
```python
model = models.resnet50(pretrained=True)
```
## Тренування з розмороженим лише останнім шаром (layer4)

### Результати трейну:
![img.png](imgs/hw4_train_resnet50_1.png)

### Результати валідації:
![img.png](data/hw4_valid_resnet50_1.png)

### Візуалізація кількох передбачень:
![img.png](imgs/hw4_images_resnet50_1.png)

### Візуалізація неправильних передбачень:
![img.png](imgs/hw4_false_images_resnet50_1.png)

### Кількість правильно помічених даних тесту:
```python
              precision    recall  f1-score   support

         dog       0.99      0.98      0.99       972
       horse       0.98      0.98      0.98       524
    elephant       1.00      1.00      1.00       285
   butterfly       0.99      0.99      0.99       435
     chicken       1.00      0.99      0.99       562
         cat       0.97      0.99      0.98       362
         cow       0.96      0.97      0.97       334
       sheep       0.98      0.98      0.98       373
      spider       0.99      0.99      0.99      1004
    squirrel       0.99      0.98      0.99       384

    accuracy                           0.99      5235
   macro avg       0.98      0.99      0.98      5235
weighted avg       0.99      0.99      0.99      5235
```

### Матриця плутанини:
![img.png](imgs/hw4_conf_matrix_resnet50_1.png)

## Тренування з розмороженням в ході трейну:

### Результати трейну:
![img.png](imgs/hw4_train_resnet50_2.png)

### Результати валідації:
![img.png](imgs/hw4_valid_resnet50_2.png)

### Візуалізація кількох передбачень:
![img.png](imgs/hw4_images_resnet50_2.png)

### Візуалізація неправильних передбачень:
![img_1.png](imgs/hw4_false_images_resnet50_2.png)

### Кількість правильно помічених даних тесту:
```python
              precision    recall  f1-score   support

         dog       0.98      0.96      0.97       972
       horse       0.95      0.96      0.96       538
    elephant       0.98      0.97      0.97       298
   butterfly       0.98      0.98      0.98       444
     chicken       0.98      0.98      0.98       597
         cat       0.98      0.96      0.97       344
         cow       0.93      0.95      0.94       380
       sheep       0.95      0.93      0.94       350
      spider       0.98      0.99      0.99       967
    squirrel       0.94      0.97      0.95       345

    accuracy                           0.97      5235
   macro avg       0.96      0.96      0.96      5235
weighted avg       0.97      0.97      0.97      5235
```

### Матриця плутанини:
![img.png](imgs/hw4_conf_matrix_resnet50_2.png)

## Тренування з розмороженням в ході трейну:

### Модель:
```python
model = models.resnet18(pretrained=True)
```

### Результати трейну:
![img.png](imgs/hw4_train_resnet18.png)

### Результати валідації:
![img.png](imgs/hw4_valid_resnet18.png)

### Візуалізація кількох передбачень:
![img.png](imgs/hw4_images_resnet18.png)

### Візуалізація неправильних передбачень:
![img.png](imgs/hw4_false_images_resnet18.png)

### Кількість правильно помічених даних тесту:
```python
              precision    recall  f1-score   support

         dog       0.97      0.96      0.97       704
       horse       0.96      0.95      0.95       365
    elephant       0.96      0.96      0.96       227
   butterfly       0.93      0.99      0.96       321
     chicken       0.98      0.96      0.97       489
         cat       0.92      0.96      0.94       254
         cow       0.96      0.93      0.94       285
       sheep       0.92      0.97      0.95       291
      spider       0.99      0.97      0.98       743
    squirrel       0.98      0.96      0.97       247

    accuracy                           0.96      3926
   macro avg       0.96      0.96      0.96      3926
weighted avg       0.96      0.96      0.96      3926
```

### Матриця плутанини:
![img.png](imgs/hw4_conf_matrix_resnet18.png)

## Тренування з розмороженням в ході трейну:

### Модель:
```python
model = models.mobilenet_v2(pretrained=True)
```

### Результати трейну:
![img.png](imgs/mn_train.png)

### Результати валідації:
![img.png](imgs/mn_valid.png)

### Візуалізація кількох передбачень:
![img.png](imgs/mn_imgs.png)

### Візуалізація неправильних передбачень:
![img.png](imgs/mn_false.png)

### Кількість правильно помічених даних тесту:
```python
              precision    recall  f1-score   support

         dog       0.91      0.96      0.94       683
       horse       0.95      0.92      0.94       386
    elephant       0.98      0.94      0.96       226
   butterfly       0.95      0.96      0.96       314
     chicken       0.99      0.96      0.97       509
         cat       0.97      0.89      0.92       249
         cow       0.92      0.90      0.91       254
       sheep       0.92      0.94      0.93       281
      spider       0.98      0.98      0.98       729
    squirrel       0.95      0.96      0.95       295

    accuracy                           0.95      3926
   macro avg       0.95      0.94      0.95      3926
weighted avg       0.95      0.95      0.95      3926
```

### Матриця плутанини:
![img.png](imgs/mn_cm.png)

## Тренування з розмороженням в ході трейну:

### Модель:
```python
model = models.resnet101(pretrained=True)
```

### Результати трейну:
![img.png](imgs/resnet101_train.png)

### Результати валідації:
![img.png](imgs/resnet101_valid.png)

### Візуалізація кількох передбачень:
![img.png](imgs/resnet101_images.png)

### Візуалізація неправильних передбачень:
![img.png](imgs/resnet101_false.png)

### Кількість правильно помічених даних тесту:
```python
              precision    recall  f1-score   support

         dog       0.99      0.95      0.97       223
       horse       0.98      1.00      0.99       220
    elephant       0.98      0.99      0.98       216
   butterfly       0.98      1.00      0.99       207
     chicken       1.00      0.97      0.98       227
         cat       0.97      0.95      0.96       216
         cow       0.93      0.97      0.95       219
       sheep       0.95      0.96      0.95       224
      spider       0.99      0.98      0.98       208
    squirrel       0.98      0.98      0.98       209

    accuracy                           0.97      2169
   macro avg       0.97      0.97      0.97      2169
weighted avg       0.97      0.97      0.97      2169 
```

### Матриця плутанини:
![img.png](imgs/resnet101_cm.png)