#ifndef VALUE_H
#define VALUE_H

#include <ostream>
#include <vector>
#include <algorithm>
#include <map>

/**
* Klasa reprezentujaca atrybut
* Przechowuje id atrybutu oraz wektor odpowiedzi wystepujacych z danym atrybutem
*/
class AttributeValue
{
private:
	/**
	* Zmienna prywatna
	* Wektor odpowiedzi wystepujacych z danym atrybutem
	*/
	std::vector<int> answers;
public:
	/**
	* Kontruktor domyslny
	*/
	AttributeValue() {}
	/**
	* Metoda publiczna
	* Zwraca wektor odpowiedzi
	* @return Wektor odpowiedzi
	*/
	const std::vector<int> &getAnswers() const
	{
		return answers;
	}
	/**
	* Metoda publiczna
	* Pobiera jeden parametr typu string
	* Dodaje nowe odpowiedzi do wektora odpowiedzi
	* @param answer Nowa odpowiedz
	*/
	void initializeAnswers()
	{
		answers.push_back(0);
	}
	/**
	* Metoda publiczna
	* Pobiera jeden parametr typu string
	* Zwieksza licznik wystapien danej odpowiedzi
	* @param answer Nowa odpowiedz
	*/
	void incrementAnswerCount(int answer_id)
	{
		++answers[answer_id];
	}
};

#endif