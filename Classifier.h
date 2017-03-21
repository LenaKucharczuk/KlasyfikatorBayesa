#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ostream>
#include <vector>
#include <algorithm>
#include "AttributeCategory.h"
#include "CUDA_functions.cuh"
using namespace std;

/**
* Klasa Klasyfikatora
* G³ówna klasa
*
* Przechowuje dane potrzebne do obliczenia odpowiednich prawdopodobienstw.
*/
class Classifier
{
private:
	typedef map<string, AttributeValue>::const_iterator atribs_iterator;

	struct AnswerStruct
	{
		int count;
		int id;
		AnswerStruct(int c = 0, int i = 0) : count(c), id(i) {}
	};
	/**
	* Zmienna prywatna.
	* Wektor wszystkich kategoriii przechowywanych przez klasyfikator.
	*/
	vector<AttributeCategory> categories;
	/**
	* Zmienna prywatna.
	* Mapa odpowiedzi klasyfikatora wraz z liczba ich wystapien
	*/
	map<string, AnswerStruct> answers;
	/**
	* Zmienna prywatna
	* Wektor zapytania uzytkownika
	*/
	vector<string> query;
	/**
	* Zmienna prywatna
	* Liczba wszystkich atrybutow
	*/
	int atribsNumber;
	/**
	* Zmienna prywatna
	* Liczba wszystkich odpowiedzi
	*/
	int answersNumber;	//liczba wszystkich odpowiedzi
						/**
						* Zmienna prywatna
						* Liczba wszystkich kategorii
						*/
	int categoriesNumber;


	/**
	* Zmienna prywatna(wykorzystywana przez procesor graficzny)
	* Tablica znakow rozpatrywanego atrybutu
	*/
	char *atribsValues;
	/**
	* Zmienna prywatna(wykorzystywana przez procesor graficzny)
	* Rozmiar tablicy znakow rozpatrywanego atrybutu
	*/
	int atribsCharNumber;
	/**
	* Zmienna prywatna(wykorzystywana przez procesor graficzny)
	*
	*/
	int *atribsPrefix; // [atribsNumber + 1];	
					   /**
					   * Zmienna prywatna(wykorzystywana przez procesor graficzny)
					   * Tablica prawdopodobienst
					   */
	double *possibilities; //[answersNumber][atribsNumber]; //tablica prawdopodobieñstw
						   /**
						   * Zmienna prywatna(wykorzystywana przez procesor graficzny)
						   * Tablica znakow z zapytania uzytkownika
						   */
	char *queryValues;	// tablica znakow z zapytania
						/**
						* Zmienna prywatna(wykorzystywana przez procesor graficzny)
						* Rozmiar tablicy znakow z zapytania uzytkownika
						*/
	int queryCharNumber;	// rozmiar tablicy
							/**
							* Zmienna prywatna(wykorzystywana przez procesor graficzny)
							*
							*/
	int *queryPrefix;	// [categoriesNumber + 1];

public:
	/**
	* Kontruktor
	* Inicjuje niektore zmienne
	*/
	Classifier()
	{
		atribsNumber = 0;
		answersNumber = 0;
		categoriesNumber = 0;
		atribsCharNumber = 0;
		queryCharNumber = 0;
		possibilities = nullptr;
		queryPrefix = nullptr;
		atribsValues = nullptr;
		atribsPrefix = nullptr;
	}
	/**
	* Destruktor
	* Usuwa zmienne dynamiczne
	*/
	~Classifier()
	{
		if (possibilities != nullptr)
			delete[] possibilities;
		if (queryPrefix != nullptr)
			delete[] queryPrefix;
		if (atribsValues != nullptr)
			delete[] atribsValues;
		if (atribsPrefix != nullptr)
			delete[] atribsPrefix;
		//cudaDeviceReset();
	}
	/**
	* Metoda publiczna
	* Pobiera wektor zmiennych typu string i inicjuje nim nastepna kategorie
	* oraz zwieksza rozmiar tablicy znakow atrybutow
	* @param values wektor atrybutow
	* @see atribsCharNumber
	*/
	void addCategory(vector<string> values)
	{
		++categoriesNumber;
		atribsNumber += values.size();
		AttributeCategory newCat;
		for (string s : values)
		{
			newCat.addValue(s);
			atribsCharNumber += s.size();
		}
		categories.push_back(newCat);
	}
	/**
	* Metoda publiczna
	* Pobiera wektor zmiennych typu string i inicjuje nimi wektor odpowiedz
	* @param newAns wektor odpowiedzi
	* @see initializerAtribsAnswers()
	*/
	void addAnswers(vector<string> newAns)
	{
		for (string a : newAns)
		{
			answers[a] = AnswerStruct(0, answersNumber++);
			for (int i = 0; i < categories.size(); ++i)
			{
				categories[i].initializeAtribsAnswers();
			}
		}
	}

	/**
	* Metoda publiczna
	* Pobiera zmienna typu string oraz wektor zmiennch typu string
	* Zwieksza licznik wystapien danej odpowiedzi
	* Dodaje przyklad trenujacy do klasyfikatora
	* @param answer zmienna przechowujaca odpowiedz na przyklad trenujacy
	* @param values wektor atrybutow w przykladzie trenujacym
	* @see incrementAnswerCount()
	*/
	void addTrainingExample(string answer, vector<string> values)
	{
		if (values.size() != categories.size()) return;
		++answers[answer].count;
		for (int i = 0; i < categories.size(); ++i)
		{
			categories[i].incrementAnswerCount(values[i], answers[answer].id);
		}
	}

	/**
	* Metoda publiczna
	* Nie pobiera argumentow
	* Zajmuje sie trenowaniem klasyfikatora na podstawie przekazanych mu przykladow trenujacych
	*/
	void train()
	{
		possibilities = new double[answersNumber*atribsNumber];

		queryPrefix = new int[categoriesNumber + 1];
		atribsValues = new char[atribsCharNumber];
		atribsPrefix = new int[atribsNumber + 1];
		atribsPrefix[0] = 0;
		int currAtrib = 0;

		for (int i = 0; i < categories.size(); ++i)	// petla po kategoriach
		{
			for (atribs_iterator it = categories[i].getAtribs().begin(); it != categories[i].getAtribs().end(); ++it)	// petla po atrybutach
			{
				it->first.copy(atribsValues + atribsPrefix[currAtrib], it->first.size());
				for (int j = 0; j < it->second.getAnswers().size(); ++j)	//petla po odpowiedziach
				{
					if (it->second.getAnswers()[j] == 0)
						possibilities[atribsNumber*j + currAtrib] = 1.0;
					else
					{
						int answer_id = j;
						map<string, AnswerStruct>::iterator ansIt = find_if(answers.begin(), answers.end(),
							[answer_id](const pair<string, AnswerStruct> & t) -> bool { return answer_id == t.second.id; });
						possibilities[atribsNumber*j + currAtrib] = (double)it->second.getAnswers()[j] / answers[ansIt->first].count;
					}
				}
				++currAtrib;
				atribsPrefix[currAtrib] = atribsPrefix[currAtrib - 1] + it->first.size();
			}
		}

	}
	/**
	* Metoda publiczna
	* Pobiera wektor atrybutow oraz zwraca odpowiedz udzielona przez klasyfikator
	* Na podstawie zapytania przekazanego przez uzytkownika znajduje odpowiedz klasyfiaktora
	* @param query wektor atrybutow w odpowiedzi
	*/
	string ask(vector<string> query)
	{

		if (query.size() != categories.size())
			return "";
		this->query = query;

		queryPrefix[0] = 0;
		int currCat = 0;
		string qV;
		for (string q : query)
		{
			queryCharNumber += q.size();
			qV += q;
			++currCat;
			queryPrefix[currCat] = queryPrefix[currCat - 1] + q.size();
		}

		queryValues = new char[queryCharNumber];
		qV.copy(queryValues, queryCharNumber);

		int answer_id = findAnswer(queryValues, atribsValues, possibilities, queryPrefix, atribsPrefix, answersNumber, categoriesNumber, atribsNumber);

		string answer;
		if (answer_id != -1)
		{
			map<string, AnswerStruct>::iterator it = find_if(answers.begin(), answers.end(),
				[answer_id](const pair<string, AnswerStruct> & t) -> bool { return answer_id == t.second.id; });
			answer = it->first;
		}
		else
			answer = "";
		delete[] queryValues;
		return answer;
	}

	string sequentialAsk(vector<string> query)
	{
		if (query.size() != categories.size())
			return "";
		this->query = query;
		string answer;
		vector<vector<double>> results;
		int answer_id = 0;
		double pos;

		for (int i = 0; i < categories.size(); ++i)	// petla po kategoriach
		{
			vector<double> resAns(answers.size());
			vector<int> answersCount = categories[i].getAnswersCountsOf(query[i]);	//szuka atrybutu

			if (answersCount.size() == 0)
				return "";
			for (map<string, AnswerStruct>::iterator ansIt = answers.begin(); ansIt != answers.end(); ++ansIt)	//petla po odpowiedziach
			{
				answer_id = ansIt->second.id;
				if (answersCount[answer_id] == 0)
					pos = 1.0;
				else
					pos = (double)answersCount[answer_id] / ansIt->second.count;

				resAns[answer_id] = pos;
			}
			results.push_back(resAns);
		}

		double max = 0.0;
		int maxId = 0;
		//wymnozenie prawdopodobienstw
		for (int i = 0; i < answersNumber; ++i) //po odpowiedziach
		{
			for (int j = 1; j < categoriesNumber; ++j) // po kategoriach
			{
				//cout <<"(" << results[j][i] << ")";
				results[0][i] *= results[j][i];
			}
			//cout << endl;
			if (max < results[0][i])
			{
				max = results[0][i];
				maxId = i;
			}
		}
		answer_id = maxId;
		map<string, AnswerStruct>::iterator it = find_if(answers.begin(), answers.end(),
			[answer_id](const pair<string, AnswerStruct> & t) -> bool { return answer_id == t.second.id; });
		return it->first;
	}

};



#endif