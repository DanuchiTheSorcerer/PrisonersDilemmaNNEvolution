import { NeuralNetwork } from "./neuralNetwork";
import { Vector, Matrix } from "./linearAlgebra";

class Player {
  private brain: NeuralNetwork;
  public score: number = 0;
  private defectionCount: number = 0;
  private totalMoves: number = 0;

  constructor() {
    this.brain = new NeuralNetwork([8, 6, 6, 6, 2]);
  }

  public mutate(): void {
    let stability: number = Math.random() * 100;
    if (stability < 10) {
      let alteredIndex = Math.floor(Math.random() * this.brain.weights.length);
      let alteredType = Math.floor(Math.random() * 2);
      let alteration: Matrix | Vector;
      if (alteredType) {
        let unaltered = this.brain.weights[alteredIndex];
        alteration = new Matrix(unaltered.rows, unaltered.columns, () => {
          return Math.random() / 10 - 0.05;
        });
        this.brain.weights[alteredIndex].add(alteration);
      } else {
        let unaltered = this.brain.biases[alteredIndex];
        alteration = new Vector(unaltered.components.length, () => {
          return Math.random() / 10 - 0.05;
        });
        this.brain.biases[alteredIndex].add(alteration);
      }
    }
  }

  public makeDecision(input: Vector): number {
    let output: Vector = this.brain.forwards(input);
    let probability: Vector = output.softmax();
    let decision = Math.random() < probability.components[0] ? 0 : 1;

    // Track defection
    this.totalMoves++;
    if (decision === 1) {
      this.defectionCount++;
    }

    return decision;
  }

  public getDefectionPercentage(): number {
    return this.totalMoves > 0 ? (this.defectionCount / this.totalMoves) * 100 : 0;
  }
}

class Game {
  public players: Player[];

  constructor() {
    this.players = [];
    for (let i = 0; i < 100; i++) {
      this.players.push(new Player());
    }
  }

  public playRound() {
    // Step 1: Match all players
    for (let i = 0; i < this.players.length; i++) {
      for (let j = i + 1; j < this.players.length; j++) {
        this.matchPlayers(i, j);
      }
    }

    // Step 2: Choose the top fifty players by score
    const topPlayers = this.players
      .slice() // Create a shallow copy to sort without mutating the original array
      .sort((a, b) => b.score - a.score) // Sort by score in descending order
      .slice(0, 50); // Take the top 50 players

    // Step 3: Display the top fifty players by score and defection percentage
    const leaderboardHtml = topPlayers
      .map((player, index) => `<div>Rank ${index + 1}: Score ${player.score} | Defection Rate: ${player.getDefectionPercentage().toFixed(2)}%</div>`)
      .join('');

    document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
      <div>
        <h1>Top 50 Players</h1>
        ${leaderboardHtml}
      </div>
    `;

    // Step 4: Replace the current population with 2 copies of the top fifty players
    this.players = [];
    for (let i = 0; i < 2; i++) {
      this.players.push(...topPlayers.map(player => {
        const newPlayer = new Player();
        newPlayer.score = 0;
        newPlayer["brain"] = player["brain"].clone(); // Assuming a `clone` method exists in NeuralNetwork
        return newPlayer;
      }));
    }

    // Step 5: Run the "mutate" method on the whole population
    this.players.forEach(player => player.mutate());
  }
  
  public matchPlayers(p1Index:number,p2Index:number) {
    let p1Moves:number[] = []
    let p2Moves:number[] = []
    // play moves
    for (let i:number = 0;i<100;i++) { //note that i is how many moves have been played
      if (i>2) {
        p1Moves[i] = this.players[p1Index].makeDecision(new Vector(8,(j:number)=>{return [p1Moves[i-1],p1Moves[i-2],p1Moves[i-3],p2Moves[i-1],p2Moves[i-2],p2Moves[i-3],i/100,Math.random()][j]}))
        p2Moves[i] = this.players[p2Index].makeDecision(new Vector(8,(j:number)=>{return [p1Moves[i-1],p1Moves[i-2],p1Moves[i-3],p2Moves[i-1],p2Moves[i-2],p2Moves[i-3],i/100,Math.random()][j]}))
      } else if (i>1) {
        p1Moves[i] = this.players[p1Index].makeDecision(new Vector(8,(j:number)=>{return [p1Moves[i-1],p1Moves[i-2],1,p2Moves[i-1],p2Moves[i-2],1,i/100,Math.random()][j]}))
        p2Moves[i] = this.players[p2Index].makeDecision(new Vector(8,(j:number)=>{return [p1Moves[i-1],p1Moves[i-2],1,p2Moves[i-1],p2Moves[i-2],1,i/100,Math.random()][j]}))
      } else if (i>0) {
        p1Moves[i] = this.players[p1Index].makeDecision(new Vector(8,(j:number)=>{return [p1Moves[i-1],1,1,p2Moves[i-1],1,1,i/100,Math.random()][j]}))
        p2Moves[i] = this.players[p2Index].makeDecision(new Vector(8,(j:number)=>{return [p1Moves[i-1],1,1,p2Moves[i-1],1,1,i/100,Math.random()][j]}))
      } else if (i==0) {
        p1Moves[i] = this.players[p1Index].makeDecision(new Vector(8,(j:number)=>{return [1,1,1,1,1,1,i/100,Math.random()][j]}))
        p2Moves[i] = this.players[p2Index].makeDecision(new Vector(8,(j:number)=>{return [1,1,1,1,1,1,i/100,Math.random()][j]}))
      }
    }
    //tally up scores and apply it
    for (let i=0;i<100;i++) {
      let move1 = p1Moves[i]
      let move2 = p2Moves[i]
      if (move1&&move2) {
        this.players[p1Index].score +=3
        this.players[p2Index].score +=3
      } else if (move1&&!move2) {
        this.players[p1Index].score += 0
        this.players[p2Index].score +=5
      } else if (!move1&&move2) {
        this.players[p1Index].score += 5
        this.players[p2Index].score += 0
      } else if (!move1&&!move2) {
        this.players[p1Index].score += 1
        this.players[p2Index].score += 1
      }
    }
  }
}

let game = new Game()
for (let i = 0; i < 1000; i++) {
  game.playRound();
}