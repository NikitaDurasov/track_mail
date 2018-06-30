import torch.utils.data as data
import numpy as np
import PIL
"""
В pytorch есть две сущности для работы с данными
1. DataSet - управляет доступом к вашим данным
2. DataLoader - инкапсулирует в себе объект DataSet и управляет процессом загрузки и вормирования пакетов (batch)
                DataSet должен переопределить функцию __getitem__ - для доступа к айтему по индексу
                                                      __len__ - для получения размера датасета
"""

class CustomC100Dataset(data.Dataset):
    """
     Для кастомных данных определяем кастомный дата сет, который позволяет загружать данные в нашем формате
     У нас два типа данных 
     1. Трейн - используем для тренировки, поскольку наш тест не размечен нам нужно отсемплировать часть нашего трейна
     2. Тест - не размеченный тест, используется для оценки качества в конкурсе. Маркеров мы тут не знаем
    """
    def __init__(self, dataset_path, dataset_type, img_cnt=50000, transform=None, target_transform=None, sample_indexies=None):
        """
         Тут мы инициализируем и загружаем данные
        :param dataset_path: - путь к файлу, который содержыт наш дата сет 
        :param dataset_type: - тип загружаемого дата сета ['test' или 'train']
        :param img_cnt: - количество картинок в датасете 50000 для трейна 10000 для теста
        :param transform: - класс трансформации для преобразования картинок
        :param target_transform: - класс транформации таргета 
        :param sample_indexies:  - интдексы для семплирования из трейна части для обучения или части для теста
        """
        if dataset_type not in ['train', 'test']:
            raise "Unknown dataset type : {}".format(dataset_type)
        self.ds_type = dataset_type
        self.ds_path = dataset_path
        self.img_cnt = img_cnt
        self.transform = transform
        self.t_transform = target_transform
        self.sample_indexies = sample_indexies
        self.__load__()


    def __load__(self):
        """
        Читаем и загружаем наши данные 
        """
        dataset = np.load(self.ds_path)
        if self.ds_type == 'train':
            dataset = dataset.reshape((self.img_cnt, 3073))
            self.y, self.x = np.hsplit(dataset, [1])
            self.y = np.amax(self.y.astype(np.int64), axis=1) # выбираем максимум из лейблов
            self.x = self.x.reshape((self.x.shape[0], 3, 32, 32))
            self.x = self.x.transpose((0, 2, 3, 1))
            if self.sample_indexies is not None:
                self.y = np.take(self.y, self.sample_indexies, axis=0)
                self.x = np.take(self.x, self.sample_indexies, axis=0)

        if self.ds_type == 'test':
            dataset = dataset.reshape((self.img_cnt, 3072))
            self.y = np.amax(np.zeros((dataset.shape[0], 1), dtype=np.int64), axis=1)
            self.x = dataset.reshape((dataset.shape[0], 3, 32, 32))
            self.x = self.x.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
        """
        Переопределяем функцию индексирования
        :param index: 
        :return: 
        """
        img, target = self.x[index], self.y[index]

        img = PIL.Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.t_transform is not None:
            target = self.t_transform(target)

        return img, target

    def __len__(self):
        """
        Переопределяем функцию получения размера DataSet
        :return: 
        """
        return len(self.x)