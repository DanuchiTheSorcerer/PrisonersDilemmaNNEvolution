import { Matrix, Vector } from './linearAlgebra.ts';

export class NeuralNetwork {
  public weights: Matrix[];

  public biases: Vector[];

  public layerSizes: number[];

  public constructor(layerSizes: number[]) {
    this.layerSizes = layerSizes;

    this.weights = [];
    this.biases = [];

    for (let i = 0; i < layerSizes.length - 1; i++) {
      const fanIn = layerSizes[i]; 
      this.weights.push(
        new Matrix(layerSizes[i + 1], layerSizes[i], () =>
          0
        )
      );
    

      this.biases.push(new Vector(layerSizes[i + 1], () => 0));
    }
  }


  public forwards(input: Vector): Vector {
    let activations = input;


    for (let i = 0; i < this.weights.length; i++) {
      activations = activations.transform(this.weights[i]).add(this.biases[i]);
      activations = this.sigmoid(activations);
    }

    return activations;
  }

  private sigmoid(input: Vector): Vector {
    return new Vector(
        input.components.length,
        (i: number) => Math.max(0, input.components[i])
    );
}

private sigmoidPrime(input: Vector): Vector {
    return new Vector(
        input.components.length,
        (i: number) => input.components[i] > 0 ? 1 : 0
    );
}


  public train(inputs: Vector[], targets: Vector[], learningRate: number): void {
    const batchSize = inputs.length;
    const weightGradients: Matrix[] = this.weights.map(
        w => new Matrix(w.rows, w.columns, () => 0)
    );
    const biasGradients: Vector[] = this.biases.map(
        b => new Vector(b.components.length, () => 0)
    );


    for (let i = 0; i < batchSize; i++) {
        const input = inputs[i];
        const target = targets[i];


        const activations: Vector[] = [];
        const zActivations: Vector[] = [];

        activations[0] = input;
        for (let j = 1; j < this.layerSizes.length; j++) {
            const z = activations[j - 1].transform(this.weights[j - 1]).add(this.biases[j - 1]);
            zActivations[j] = z;
            activations[j] = this.sigmoid(z);
        }

        let delta = activations[this.layerSizes.length - 1]
            .subtract(target)
            .mult(this.sigmoidPrime(zActivations[this.layerSizes.length - 1]));

        for (let j = this.layerSizes.length - 2; j >= 0; j--) {
            const weightGradient = new Matrix(
                this.layerSizes[j + 1],
                this.layerSizes[j],
                (row, col) => delta.components[row] * activations[j].components[col]
            );
            weightGradients[j] = weightGradients[j].add(weightGradient);

            biasGradients[j] = biasGradients[j].add(delta);

            if (j > 0) {
                delta = delta.transform(this.weights[j].transpose())
                    .mult(this.sigmoidPrime(zActivations[j]));
            }
        }
    }

    for (let j = 0; j < this.weights.length; j++) {
        weightGradients[j] = weightGradients[j].scale(1 / batchSize);
        biasGradients[j] = biasGradients[j].scale(1 / batchSize);

        this.weights[j] = this.weights[j].subtract(weightGradients[j].scale(learningRate));
        this.biases[j] = this.biases[j].subtract(biasGradients[j].scale(learningRate));
    }
}
public clone(): NeuralNetwork {
  const copy = new NeuralNetwork(this.layerSizes); // Assuming the layers structure is reusable
  copy.weights = this.weights.map(w => w.clone()); // Deep clone each weight matrix
  copy.biases = this.biases.map(b => b.clone());   // Deep clone each bias vector
  return copy;
}
}
