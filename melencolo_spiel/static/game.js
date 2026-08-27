const config = {
    type: Phaser.AUTO,
    width: 800,
    height: 600,
    parent: game,
    backgroundColor: '#b3e5fc',
    physics: {
        default: 'arcade',
        arcade: { gravity: { y: 50 }, debug: false }
    },
    scene: { preload, create, update }
};

new Phaser.Game(config);

function preload() {
    // --- Create procedural textures ---
    const jarGraphics = this.make.graphics({ x: 0, y: 0, add: false });
    jarGraphics.fillStyle(0x795548, 1);
    jarGraphics.fillRect(0, 0, 40, 60);
    jarGraphics.generateTexture('jar', 40, 60);


    const chuteGraphics = this.make.graphics({ x: 0, y: 0, add: false });
    chuteGraphics.fillStyle(0xffffff, 1);
    chuteGraphics.beginPath();
    chuteGraphics.arc(50, 50, 50, Math.PI, 2 * Math.PI, false);
    chuteGraphics.fillPath();
    chuteGraphics.generateTexture('parachute', 100, 60);

    const platform = this.make.graphics({ x: 0, y: 0, add: false });
    platform.fillStyle(0x00FF00, 1);
    platform.fillRect(0, 0, 800, 40);
    platform.generateTexture('platform', 800, 40);

    const lift = this.make.graphics({ x: 0, y: 0, add: false });
    lift.fillStyle(0x00FF00, 1);
    lift.fillRect(0, 0, 100, 40);
    lift.generateTexture('lift', 100, 40);

    const sittingMelencolo = document.getElementById("sittingMelencolo").src;
    this.load.image("sittingMelencolo", sittingMelencolo);

    const firingMelencolo = document.getElementById("firingMelencolo").src;
    this.load.image("firingMelencolo", firingMelencolo);

    const basketBottom = document.getElementById("basketBottom").src;
    this.load.image("basketBottom", basketBottom);

    const basketRim = document.getElementById("basketRim").src;
    this.load.image("basketRim", basketRim);

    const landingSpot = Phaser.Math.Between(450, 500)
}

async function create() {
    this.score = 0;

    this.scoreText = this.add.text(20, 20, "Score: 0", {
        fontSize: "32px",
        color: "#000",
        fontFamily: "Arial",
    });
    this.scoreText.setDepth(100); // stay on top

    this.platform = this.physics.add.staticImage(400, 560, 'platform');
    this.lift = this.physics.add.staticImage(580, 520, 'lift')
    this.jars = this.physics.add.group();
    this.words = await loadWords();
    console.log(this.words)


    this.basketBottom = this.add.image(580, 500, "basketBottom");
    this.basketBottom.setDepth(5);

    this.basketRim = this.add.image(580, 500, "basketRim");
    this.basketRim.setDepth(15);

    this.dragon = this.add.image(150, 450, "sittingMelencolo");

    window.gameScene = this;

}

function startGame() {
    const scene = window.gameScene;

    if (!scene) {
        console.error("Phaser scene is not ready yet.");
        return;
    }

    if (scene.gameTimer) {
        console.log("Game is already running.")
    }

    // Fire every 4 seconds
    scene.gameTimer = scene.time.addEvent({
        delay: 4000,
        loop: true,
        callback: () => {
            // switch to firing frame
            scene.dragon.setTexture("firingMelencolo");

            spawnJar(scene);

            // after 1 second go back to sitting
            scene.time.delayedCall(1000, () => {
                scene.dragon.setTexture("sittingMelencolo");
            });
        }
    });

    console.log("Game started!")
}

function spawnJar(scene) {
    const word = Phaser.Utils.Array.GetRandom(scene.words);
    const startX = 180;
    const startY = 500;

    const jar = scene.add.text(startX, startY, '🔥', {
        fontSize: "60px",
        fontFamily: "Arial"
    });
    scene.physics.add.existing(jar);
    scene.jars.add(jar);
    jar.body.setBounce(0.1);
    jar.setInteractive({ useHandCursor: true });
    jar.setDepth(20);
    jar.wordData = word;

    const label = scene.add.text(jar.x - 18, jar.y - 10, word.de, { fontSize: '36px', color: '#fff' });
    jar.label = label;
    jar.initialVX = jar.body.setVelocityX(Phaser.Math.Between(45, 50)); //horizontal push
    jar.initialVY = jar.body.setVelocityY(-200); //upward throw

    jar.on('pointerdown', () => showParachute(scene, jar));
}

function showParachute(scene, jar) {
    if (jar.parachuteOpen) return;
    jar.parachuteOpen = true;
    // Pause only THIS ONE jar
    jar.body.enable = false;
    const parachute = scene.add.image(jar.x, jar.y - 70, 'parachute');
    jar.parachute = parachute;

    const options = Phaser.Utils.Array.Shuffle(jar.wordData.en);
    jar.optionTexts = options.map((opt, i) => {
        const text = scene.add.text(
            jar.x - 60, jar.y - 120 + i * 30, opt,
            { fontSize: '18px', color: '#000', backgroundColor: '#fff', padding: 5 }
        ).setInteractive({ useHandCursor: true });

        text.on('pointerdown', () => {
            handleChoice(scene, jar, opt);
        });
        return text;
    });
}

function handleChoice(scene, jar, opt) {
    let correct = false;
    jar.optionTexts.forEach(t => {
        t.disableInteractive();
        if (opt === jar.wordData.correct && opt === t.text) {
            t.setColor('#2e7d32'); // green for correct
            correct = true;


        }
        else if (opt !== jar.wordData.correct && opt === t.text) t.setColor('#c62828'); // red for others
    })

    scene.time.delayedCall(600, () => {
        // Clean up parachute and options
        jar.optionTexts.forEach(t => t.destroy());
        jar.parachute?.destroy();
        jar.body.enable = true;
        // Apply outcome
        jar.setText(correct ? "😊" : "💣");
        if (correct) {
            scene.score += 10;
            jar.setDepth(10);
            jar.platformCollider = scene.physics.add.collider(jar, scene.lift, () => {
                jar.body.setVelocityX(0);
                jar.body.setGravity(0);
            });
            jar.body.setVelocityX(40)
        }
        else {
            /*
            If you want to obtain a sharp diagonal fall in case of wrong answer
            you need to adjust the vX and vY vectors.
            When they have the same value, the rectilinear trajectory has a 45 degrees slope.
            That means the ratio jar.x/jar.y=1.
            */
            //if (jar.x < 450) {
            scene.score -= 10;
            const tx = 580; // basket center X
            const ty = 500; // basket center Y

            const x0 = jar.x;
            const y0 = jar.y;

            const dx = tx - x0;
            const dy = ty - y0;

            // Normalize
            const len = Math.sqrt(dx * dx + dy * dy);
            const nx = dx / len;
            const ny = dy / len;

            // Choose the speed (tune this!)
            const speed = 400;

            // Apply velocity toward basket
            jar.body.setVelocity(nx * speed, ny * speed);
            //}

            // else {
            // jar.body.setVelocityY(300);
            //}
        }


        scene.scoreText.setText("Score: " + scene.score);
    });
}

function update() {
    this.jars.children.iterate(jar => {
        if (!jar || !jar.label) return;

        jar.label.x = jar.x - 18;
        jar.label.y = jar.y - 10;

        if (jar.parachute) {
            jar.parachute.x = jar.x;
            jar.parachute.y = jar.y - 70;
        }

        if (jar.y > 560 && jar.body.velocity.y < 100) {
            jar.body.setVelocityY(0);
            jar.body.allowGravity = false;
        }

        if (jar.y > 620) {
            jar.label.destroy();
            jar.destroy();
        }
    });

}

async function loadWords() {

    const res = await fetch("/game_words");
    //console.log("Pairs: "+res.json());
    return await res.json();

}