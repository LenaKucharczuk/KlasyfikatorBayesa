#ifndef CATEGORY_H
#define CATEGORY_H

#include <ostream>
#include <map>
#include <iterator>
#include "AttributeValue.h"

/**
* Klasa reprezentujaca kategorie
* Przechowuje atrybuty nalezaca do danej kategorii
*/

class AttributeCategory
{
private:
	typedef std::map<std::string, AttributeValue>::iterator atribs_iterator;
	/**
	* Zmienna prywatna
	* Mapa nazwy atrybutu oraz jego wartosci
	*/
	std::map<std::string, AttributeValue> atribs;
public:
	/**
	* Konstruktor domyslny
	*/
	AttributeCategory() {};
	/**
	* Metoda publiczna
	* Pobiera zmienna string oraz zmienna int
	* Dodaje Nowy atrybut do kategorii
	* @param newValue Nazwa atrybutu
	* @param atrib_id id nowego atrybutu
	*/
	void addValue(std::string newValue)
	{
		atribs[newValue] = AttributeValue();
	}

	/**
	* Metoda publiczna
	* Nie pobiera atrybutow i  zwraca mape
	* @return Para nazwa atrybutu i jego wartosci
	* @see atribs
	*/
	const std::map<std::string, AttributeValue>& getAtribs() const
	{
		return atribs;
	}

	/**
	* Metoda publiczna
	* Pobiera zmienna typu string
	* Wywoluje metode publiczna klasy AttributeValue i dodaje do
	* znajdujacego sie w tej klasie wektora nowa odpowiedz
	* @param answer Odpowiedz klasyfikator
	* @see AttributeValue
	* @see initializeAnswers()
	*/
	void initializeAtribsAnswers()
	{
		for (atribs_iterator it = atribs.begin(); it != atribs.end(); ++it)
		{
			it->second.initializeAnswers();
		}
	}
	/**
	* Metoda publicza
	* Pobiera dwie zmienne typu string
	* Wywoluje metode publiczna klasy AttributeValue, ktora zwieksza licznik wystapien
	* danej odpowiedzi dla konkretnego atrybutu
	* @param atrib Nazwa atrybutu
	* @param answer Odpowiedz klasyfikatora
	* @see AttributeValue
	* @see incrementAnswerCount()
	*/
	void incrementAnswerCount(std::string atrib, int answer_id)
	{
		atribs_iterator it = atribs.find(atrib);
		if (it != atribs.end())
			it->second.incrementAnswerCount(answer_id);
	}

	std::vector<int> getAnswersCountsOf(std::string atrib)
	{
		atribs_iterator it = atribs.find(atrib);
		if (it != atribs.end())
			return atribs[atrib].getAnswers();
		else
			return std::vector<int>();
	}
};

#endif