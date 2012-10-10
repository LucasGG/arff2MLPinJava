/*
MLP em Java por Phil Brierley
www.philbrierley.com
(Código modificado por Lucas Guimarães Gonçalves)

Este código pode ser usado e modificado gratuitamente.

Todos os métodos e variáveis que podem ser editados estão marcados.

- A função de ativação é tanh.
- A saída do neurônio é linear.

 */

import java.lang.Math;

public class MLPemJava {
	// TODO Variáveis definidas pelo usuário.
	static int numCiclos = 500; // Número de ciclos (épocas) de treinamento.
	static int numEntradas = 3; // Número de entradas.
	static int numUocultas = 4; // Número de neurônios ocultos.
	static int numExemplos = 4; // Número de exemplos para treinamento.
	static double LR_IH = 0.7; // Taxa de treinamento para entrada.
	static double LR_HO = 0.07; // Taxa de treinamento para saída.

	// Variáveis para processamento.
	static int numExemplo;
	static double erroExAtual;
	static double outPred;
	static double RMSerror;
	static double weightChange;

	// Matriz e vetor para conjunto de dados para entrada e saída.
	static double[][] trainInputs = new double[numExemplos][numEntradas];
	static double[] trainOutput = new double[numExemplos];

	// As saídas dos neurônios ocultos.
	static double[] hiddenVal = new double[numUocultas];

	// Os pesos de entrada e saída.
	static double[][] pesosEntrada = new double[numEntradas][numUocultas];
	static double[] pesosSaida = new double[numUocultas];

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// Inicia os pesos de entrada e saída.
		iniciarPesos();

		// Carrega os dados.
		iniciarDados();

		// Treina a rede neural.
		for (int j = 0; j <= numCiclos; j++) {
			for (int i = 0; i < numExemplos; i++) {
				// Seleciona um exemplo randômico.
				numExemplo = (int) ((Math.random() * numExemplos) - 0.001);

				/*
				 * Calcula a saída da rede neural atual e o erro para esse
				 * exemplo.
				 */
				calcRede();

				// Recalcula os pesos.
				WeightChangesHO();
				WeightChangesIH();
			}

			// Calcula o erro geral da rede após cada época.
			calcOverallError();
			System.out.println("Ciclo: " + j + "\tErro RMS = " + RMSerror);
		}

		// Mostra todos os resultados.
		mostrarResultados();
	}

	static void iniciarPesos() {
		System.out.println("Inicializando pesos...");

		for (int j = 0; j < numUocultas; j++) {
			pesosSaida[j] = (Math.random() - 0.5) / 2;
			for (int i = 0; i < numEntradas; i++)
				pesosEntrada[i][j] = (Math.random() - 0.5) / 5;
		}
	}

	static void iniciarDados() {
		System.out.println("Inicializando dados...");

		// TODO Coloque os exemplos aqui.

		// Ex.: Uma operação lógica XOR.
		trainInputs[0][0] = 1;
		trainInputs[0][1] = -1;
		trainInputs[0][2] = 1; // bias
		trainOutput[0] = 1;

		trainInputs[1][0] = -1;
		trainInputs[1][1] = 1;
		trainInputs[1][2] = 1; // bias
		trainOutput[1] = 1;

		trainInputs[2][0] = 1;
		trainInputs[2][1] = 1;
		trainInputs[2][2] = 1; // bias
		trainOutput[2] = -1;

		trainInputs[3][0] = -1;
		trainInputs[3][1] = -1;
		trainInputs[3][2] = 1; // bias
		trainOutput[3] = -1;
	}

	static void calcRede() {
		// Calcula a saída dos neurônios ocultos.
		for (int i = 0; i < numUocultas; i++) {
			hiddenVal[i] = 0.0;

			for (int j = 0; j < numEntradas; j++)
				hiddenVal[i] += (trainInputs[numExemplo][j] * pesosEntrada[j][i]);

			hiddenVal[i] = Math.tanh(hiddenVal[i]);
			
			// Regularização da entrada.
			if (hiddenVal[i] > 20)
				hiddenVal[i] = 1;
			else if (hiddenVal[i] < -20)
				hiddenVal[i] = -1;
		}

		// Calcula a saída da rede.
		// A saída do neurônio é linear.
		outPred = 0.0;

		for (int i = 0; i < numUocultas; i++)
			outPred = outPred + hiddenVal[i] * pesosSaida[i];

		// Calcula o erro.
		erroExAtual = outPred - trainOutput[numExemplo];
	}

	static void WeightChangesHO() {
		for (int i = 0; i < numUocultas; i++) {
			weightChange = LR_HO * erroExAtual * hiddenVal[i];
			pesosSaida[i] -= weightChange;

			// Regularização da saída.
			if (pesosSaida[i] < -5)
				pesosSaida[i] = -5;
			else if (pesosSaida[i] > 5)
				pesosSaida[i] = 5;
		}
	}

	static void WeightChangesIH() {
		for (int i = 0; i < numUocultas; i++)
			for (int j = 0; j < numEntradas; j++) {
				weightChange = 1 - (hiddenVal[i] * hiddenVal[i]);
				weightChange *= pesosSaida[i] * erroExAtual * LR_IH
						* trainInputs[numExemplo][j];
				pesosEntrada[j][i] -= weightChange;
			}
	}

	static void calcOverallError() {
		RMSerror = 0.0;

		for (int i = 0; i < numExemplos; i++) {
			numExemplo = i;
			calcRede();
			RMSerror += (erroExAtual * erroExAtual);
		}

		RMSerror = Math.sqrt(RMSerror / numExemplos);
	}

	static void mostrarResultados() {
		for (int i = 0; i < numExemplos; i++) {
			numExemplo = i;
			calcRede();

			System.out.println("Exemplo = " + (numExemplo + 1) + "\tactual = "
					+ trainOutput[numExemplo] + "\tneural model = " + outPred);
		}
	}
}
