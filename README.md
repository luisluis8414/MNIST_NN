# Perceptron

Dieses Projekt ist eine Implementierung eines Perceptrons in C++. Ein Perceptron ist ein einfacher, linearer Klassifikator, der Datenpunkte in zwei Klassen einteilen kann. Er basiert auf der Idee, dass die Daten durch eine lineare Entscheidungsgrenze (z. B. eine gerade Linie in 2D) getrennt werden können.

## Das XOR Problem

Versuchen wir unser Pereptron mit dem XOR-Problem zu trainieren:

```cpp
std::vector<int> training_outputs = {1, 0, 0, 1};
```

so fällt auf, das der Output ein falsches Ergebnis liefert:

```bash
0 AND 0 = 0
0 AND 1 = 0
1 AND 0 = 1
1 AND 1 = 1
```

Das liegt daran, dass die Wahrheitstabelle eines XOR-Gatters nicht linear separabel ist.
