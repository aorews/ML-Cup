## Решение онлайн части соревнования Yandex Cup ML Challenge 2021 NLP

Ограничения на выполнение:

1 ядро CPU

8 Gb оперативной памяти

20 минут времени на исполнение run.sh и скрипта оценивания score.py вместе взятых

700 MB максимальный объем архива с решением

Исходя из этого был использован beam search, во время которого подбирались ближайшие по косинусному расстоянию замены из словарей с предподсчитанными токсичностями.
