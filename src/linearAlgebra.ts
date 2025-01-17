export class Vector {
  public components: number[] = [];
  public constructor(components: number, genFunc: Function) {
    this.components = [];
    for (let i = 0; i < components; i++) {
      this.components[i] = genFunc(i);
    }
  }
  public add(otherVector: Vector): Vector {
    let newComponents: number[] = [];
    for (let i = 0; i < this.components.length; i++) {
      newComponents[i] = otherVector.components[i] + this.components[i];
    }

    return new Vector(this.components.length, (i: number) => newComponents[i]);
  }
  public mult(otherVector: Vector): Vector {
    let newComponents: number[] = [];
    for (let i = 0; i < this.components.length; i++) {
      newComponents[i] = otherVector.components[i] * this.components[i];
    }

    return new Vector(this.components.length, (i: number) => newComponents[i]);
  }
  public scale(scalar: number): Vector {
    let newComponents: number[] = [];
    for (let i = 0; i < this.components.length; i++) {
      newComponents[i] = this.components[i]*scalar;
    }

    return new Vector(this.components.length, (i: number) => newComponents[i]);
  }
  public subtract(otherVector: Vector): Vector {
    let newComponents: number[] = [];
    for (let i = 0; i < this.components.length; i++) {
      newComponents[i] = this.components[i] - otherVector.components[i];
    }

    return new Vector(this.components.length, (i: number) => newComponents[i]);
  }
  public transform(transformer: Matrix): Vector {

    let newComponents: number[] = [];

    for (let i = 0; i < transformer.rows; i++) {
      let sum = 0;
      for (let j = 0; j < transformer.columns; j++) {
        sum += transformer.matrix[i][j] * this.components[j];
      }
      newComponents[i] = sum;
    }

    return new Vector(transformer.rows, (i: number) => newComponents[i]);
  }
  public softmax(): Vector {
    const expComponents = this.components.map((value) => Math.exp(value));

    const sumExp = expComponents.reduce((sum, value) => sum + value, 0);

    const softmaxed = expComponents.map((value) => value / sumExp);

    return new Vector(this.components.length, (i:number) => softmaxed[i]);
  }
  public clone(): Vector {
    return new Vector(this.components.length, (i:number) => this.components[i]);
  }
}

export class Matrix {
  public matrix: number[][] = [];
  public rows: number;
  public columns: number;

  public constructor(
    rows: number,
    columns: number,
    genFunc: (row: number, col: number) => number
  ) {
    this.rows = rows;
    this.columns = columns;
    for (let i = 0; i < rows; i++) {
      this.matrix[i] = [];
      for (let j = 0; j < columns; j++) {
        this.matrix[i][j] = genFunc(i, j);
      }
    }
  }

  public transpose(): Matrix {
    return new Matrix(this.columns, this.rows, (row, col) => this.matrix[col][row]);
  }

  public scale(scalar: number): Matrix {
    return new Matrix(this.rows, this.columns, (row, col) => this.matrix[row][col] * scalar);
  }

  public subtract(other: Matrix): Matrix {
    return new Matrix(this.rows, this.columns, (row, col) => this.matrix[row][col] - other.matrix[row][col]);
  }
  public add(other: Matrix): Matrix {
    return new Matrix(this.rows, this.columns, (row, col) => this.matrix[row][col] + other.matrix[row][col]);
  }
  public clone(): Matrix {
    return new Matrix(this.rows, this.columns, (i, j) => this.matrix[i][j]);
  }
}
