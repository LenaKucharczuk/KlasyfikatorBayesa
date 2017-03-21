#include "CUDA_functions.cuh"
#include "CUDA_exception.h"
#include "CUDA_malloc_exception.h"
#include "CUDA_memcpy_exception.h"
#include "CUDA_device_exception.h"
#include "CUDA_kernel_exception.h"
#include "CUDA_synchronize_exception.h"
#include "CUDA_error.h"

int answersNumber;
int categoriesNumber;
int atribsNumber;

/**
* Funkcja wykonywana na karcie graficznej - kazdy watek sprawdza czy jego atrybut z atribsValues to ten sam co w query. Jesli tak, przepisuje do
* tablicy wynikowej prawdopodobiestwa dla kazdej jego odpowiedzi
* @param query - zapytanie uzytkownika w postacie zlepionych stringow
* @param atribsValues - tablica wszystkich atrybutow
* @param possibilities - tablica wszystkich prawdopodobienstw
* @param queryPrefix - tablica sum prefiksowych dlugosci slow w query
* @param atribsPrefix - j.k. dla atribsValues
* @param answersNumber - liczba mozliwych odpowiedzi
* @param categoriesNumber - liczba kategorii
* @param atribsNumber - liczba wszystkich atrybutow
* @param resultPossibilities - tablica prawdopodobienstw atrybutow z zapytania dla wszystkich mozliwych odpowiedzi
*/
__global__ void searchWithCuda(double *resultPossibilities, char *query, char *atribsValues, double *possibilities, int *queryPrefix, int *atribsPrefix, int *answersNumber, int *categoriesNumber, int *atribsNumber)
{
	int category_id = blockIdx.x;	// categories
	int atrib_id = blockIdx.y;	// atribs

								// znajdz poczatek lancucha znakow atrybutu w zapytaniu i w atribsValue
	char *queryAtrib = query + queryPrefix[category_id];
	int queryAtribLength = queryPrefix[category_id + 1] - queryPrefix[category_id];

	char *currAtrib = atribsValues + atribsPrefix[atrib_id];
	int currAtribLength = atribsPrefix[atrib_id + 1] - atribsPrefix[atrib_id];

	if (queryAtribLength == currAtribLength)
	{
		bool equal = true;
		for (int i = 0; i < queryAtribLength; ++i)
		{
			if (queryAtrib[i] != currAtrib[i])
			{
				equal = false;
				break;
			}
		}
		if (equal)	// przypisz odpowiednie prawdopodobienstwa
		{
			for (int i = 0; i < *answersNumber; ++i)
			{
				resultPossibilities[*categoriesNumber*i + category_id] = possibilities[*atribsNumber*i + atrib_id];	// na razie tylko dla jednej odpowiedzi
			}
		}
	}
}

void setDevice()
{
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
		throw CUDA_device_exception();
}

void getLastError()
{
	cudaError_t code;
	if ((code = cudaGetLastError()) != cudaSuccess)
		throw CUDA_error(code);
}

void deviceSynchronize()
{
	if (cudaDeviceSynchronize() != cudaSuccess)
		throw CUDA_synchronize_exception();
}

/**
* Funkcja wywolujaca funkcje dzialajaca na GPU. Po uzyskaniu wynikowej tablicy prawdopodobienstw wymnaza prawdopodobienstwa dla kazdej odpowiedzi i
* wybiera najlepiej dopasowana
* @param query - zapytanie uzytkownika w postacie zlepionych stringow
* @param atribsValues - tablica wszystkich atrybutow
* @param possibilities - tablica wszystkich prawdopodobienstw
* @param queryPrefix - tablica sum prefiksowych dlugosci slow w query
* @param atribsPrefix - j.k. dla atribsValues
* @param answersNumber - liczba mozliwych odpowiedzi
* @param categoriesNumber - liczba kategorii
* @param atribsNumber - liczba wszystkich atrybutow
* @return indeks najlepiej dopasowanej odpowiedzi
*/
__host__ int findAnswer(char *query, char *atribsValues, double *possibilities, int *queryPrefix, int *atribsPrefix, int answersNumber, int categoriesNumber, int atribsNumber)
{
	int queryCharNumber = queryPrefix[categoriesNumber];
	int atribsCharNumber = atribsPrefix[atribsNumber];

	double* resultPossibilities = new double[answersNumber*categoriesNumber];

	double *dev_resultPossibilities = 0;
	char *dev_query = 0;
	char *dev_atribsValues = 0;
	double *dev_possibilities = 0;
	int *dev_queryPrefix = 0;
	int *dev_atribsPrefix = 0;
	int *dev_answersNumber = 0;
	int *dev_categoriesNumber = 0;
	int *dev_atribsNumber = 0;

	try
	{
		setDevice();

		if (cudaMalloc((void**)&dev_resultPossibilities, answersNumber * categoriesNumber * sizeof(double)) != cudaSuccess)
			throw CUDA_malloc_exception();
		if (cudaMalloc((void**)&dev_query, queryCharNumber * sizeof(char)) != cudaSuccess)
			throw CUDA_malloc_exception();
		if (cudaMalloc((void**)&dev_atribsValues, atribsCharNumber * sizeof(char)) != cudaSuccess)
			throw CUDA_malloc_exception();
		if (cudaMalloc((void**)&dev_possibilities, answersNumber * atribsNumber * sizeof(double)) != cudaSuccess)
			throw CUDA_malloc_exception();
		if (cudaMalloc((void**)&dev_queryPrefix, (categoriesNumber + 1) * sizeof(int)) != cudaSuccess)
			throw CUDA_malloc_exception();
		if (cudaMalloc((void**)&dev_atribsPrefix, (atribsNumber + 1) * sizeof(int)) != cudaSuccess)
			throw CUDA_malloc_exception();
		if (cudaMalloc((void**)&dev_answersNumber, sizeof(int)) != cudaSuccess)
			throw CUDA_malloc_exception();
		if (cudaMalloc((void**)&dev_categoriesNumber, sizeof(int)) != cudaSuccess)
			throw CUDA_malloc_exception();
		if (cudaMalloc((void**)&dev_atribsNumber, sizeof(int)) != cudaSuccess)
			throw CUDA_malloc_exception();

		if (cudaMemcpy(dev_query, query, queryCharNumber * sizeof(char), cudaMemcpyHostToDevice) != cudaSuccess)
			throw CUDA_memcpy_exception();
		if (cudaMemcpy(dev_atribsValues, atribsValues, atribsCharNumber * sizeof(char), cudaMemcpyHostToDevice) != cudaSuccess)
			throw CUDA_memcpy_exception();
		if (cudaMemcpy(dev_possibilities, possibilities, answersNumber * atribsNumber * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
			throw CUDA_memcpy_exception();
		if (cudaMemcpy(dev_queryPrefix, queryPrefix, (categoriesNumber + 1) * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
			throw CUDA_memcpy_exception();
		if (cudaMemcpy(dev_atribsPrefix, atribsPrefix, (atribsNumber + 1) * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
			throw CUDA_memcpy_exception();
		if (cudaMemcpy(dev_answersNumber, &answersNumber, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
			throw CUDA_memcpy_exception();
		if (cudaMemcpy(dev_categoriesNumber, &categoriesNumber, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
			throw CUDA_memcpy_exception();
		if (cudaMemcpy(dev_atribsNumber, &atribsNumber, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
			throw CUDA_memcpy_exception();


		dim3 dimBlock(categoriesNumber, atribsNumber, 1); // x, y, z
		searchWithCuda << <dimBlock, 1 >> >(dev_resultPossibilities, dev_query, dev_atribsValues, dev_possibilities, dev_queryPrefix, dev_atribsPrefix, dev_answersNumber, dev_categoriesNumber, dev_atribsNumber);

		getLastError();
		deviceSynchronize();


		if (cudaMemcpy(resultPossibilities, dev_resultPossibilities, answersNumber * categoriesNumber * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
			throw CUDA_memcpy_exception();
	}
	catch (exception& e)
	{
		printf("%s", e.what());
		cudaFree(dev_atribsPrefix);
		cudaFree(dev_atribsValues);
		cudaFree(dev_possibilities);
		cudaFree(dev_query);
		cudaFree(dev_queryPrefix);
		cudaFree(dev_resultPossibilities);
		cudaFree(dev_answersNumber);
		cudaFree(dev_categoriesNumber);
		cudaFree(dev_atribsNumber);
		delete[] resultPossibilities;

		return -1;
	}

	cudaFree(dev_atribsPrefix);
	cudaFree(dev_atribsValues);
	cudaFree(dev_possibilities);
	cudaFree(dev_query);
	cudaFree(dev_queryPrefix);
	cudaFree(dev_resultPossibilities);
	cudaFree(dev_answersNumber);
	cudaFree(dev_categoriesNumber);
	cudaFree(dev_atribsNumber);


	double *answersPos = new double[answersNumber];
	double max = 0.0;
	int maxId = 0;
	for (int i = 0; i < answersNumber; ++i)
	{
		answersPos[i] = 1;
		for (int j = 0; j < categoriesNumber; ++j)
			answersPos[i] *= resultPossibilities[i*categoriesNumber + j];
		if (answersPos[i] > max)
		{
			maxId = i;
			max = answersPos[i];
		}
	}


	delete[] resultPossibilities;
	delete[] answersPos;
	return maxId;
}