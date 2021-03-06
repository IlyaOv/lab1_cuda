**cuda_lab1.cu** - код программы.<br/>
**CUDA_lab1_gc.ipynb** - ipynb файл Google Colab, где производились расчеты.<br/>
<br/>
В **первом** случае (**matrixMult**) каждая нить вычисляет 1 элемент итоговой матрицы C. По индексам блока и нити можно определить индексы начала строки матрицы A и столбца матрицы B, а также индекс элемента матрицы C, куда и будет записан результат работы нити. Минусом данного подхода является то, что элемнты матриц A и B подгружаются N раз (размерность квадратных матриц) из глобальной памяти.<br/>
Во **втором** случае (**matrixMultShared**) используется алгоритм с **разделяемой памятью**. Подматрицы матриц A и B подгружаются в разделяемую память (__shared__) и каждая нить загружает один элемент каждой матрицы. Затем каждая нить вычисляет один элемент блочной подматрицы. В результате 1 нить записвает 1 элемент в глобальную память. __syncthreads() используется для синхронизации загрузки матриц в разделяемую память и вычислений над ними. <br/>
<br/>
Результаты времени выполнения программ для двух случаев приведены в таблицах ниже.

### Время работы и ускорение параллельного алгоритма
| Размерность матрицы | 112 |  512 | 1008 | 1504 | 2000 |
|:----:|:----:|:----:|:----:|:----:|:----:|
|**Время работы <br /> алгоритма на CUDA, мс.**| 0,066 | 3,04 | 22,832 | 75,304 | 177,183 |
|**Время работы <br /> последовательного алгоритма, мс.**| 6,33 | 1435,023 | 6522,139 | 28814,307 | 89697,769 |
|**Ускорение, раз**| 95,234 | 471,938 | 285,65 | 382,641 | 506,243 |


### Время работы и ускорение параллельного алгоритма на CUDA с разделяемой памятью (shared)
| Размерность матрицы | 112 |  512 | 1008 | 1504 | 2000 |
|:----:|:----:|:----:|:----:|:----:|:----:|
|**Время работы <br /> алгоритма на CUDA, мс.**| 0,069 | 3,01 | 22,619 | 74,52 | 170,708 |
|**Время работы <br /> последовательного алгоритма, мс.**| 7,163 | 1387,854 | 6379,626 | 30183,119 | 100188,594 |
|**Ускорение, раз**| 103,823 | 461,103 | 282,046 | 405,034 | 586,899 |
<br/>
Из полученных результатов видно, что наибольшее ускорение параллельного алгоритма достигается при размерности матриц 2000 на 2000 эл. При использовании разделяемой памяти выигрыш по времени наблюдается при размерности матриц 512 на 512 элементов и больше. 
