/*RESPONSIVE MELENCOLO*/
/* ================================================================
   PHASER CONFIGURATION
   ================================================================ */

const config = {

    type: Phaser.AUTO,

    /*
        Keep the game's internal coordinate system at 800 × 600.

        Phaser will scale this to fit the available screen.
    */
    width: 800,
    height: 600,

    parent: "game",

    backgroundColor: "#b3e5fc",

    /*
        RESPONSIVE SCALING
    */
    scale: {

        /*
            Scale the 800×600 game to fit its container
            while preserving its aspect ratio.
        */
        mode: Phaser.Scale.FIT,

        /*
            Keep the game centered inside the available area.
        */
        autoCenter: Phaser.Scale.CENTER_BOTH,

        width: 800,
        height: 600
    },

    physics: {

        default: "arcade",

        arcade: {
            gravity: {
                y: 50
            },

            debug: false
        }
    },

    scene: {
        preload,
        create,
        update
    }
};


new Phaser.Game(config);


/* ================================================================
   PRELOAD
   ================================================================ */

function preload() {

    /* ------------------------------------------------------------
       Jar texture
       ------------------------------------------------------------ */

    const jarGraphics = this.make.graphics({
        x: 0,
        y: 0,
        add: false
    });

    jarGraphics.fillStyle(0x795548, 1);

    jarGraphics.fillRect(
        0,
        0,
        40,
        60
    );

    jarGraphics.generateTexture(
        "jar",
        40,
        60
    );


    /* ------------------------------------------------------------
       Parachute texture
       ------------------------------------------------------------ */

    const chuteGraphics = this.make.graphics({
        x: 0,
        y: 0,
        add: false
    });

    chuteGraphics.fillStyle(0xffffff, 1);

    chuteGraphics.beginPath();

    chuteGraphics.arc(
        50,
        50,
        50,
        Math.PI,
        2 * Math.PI,
        false
    );

    chuteGraphics.fillPath();

    chuteGraphics.generateTexture(
        "parachute",
        100,
        60
    );


    /* ------------------------------------------------------------
       Platform texture
       ------------------------------------------------------------ */

    const platform = this.make.graphics({
        x: 0,
        y: 0,
        add: false
    });

    platform.fillStyle(0x00ff00, 1);

    platform.fillRect(
        0,
        0,
        800,
        40
    );

    platform.generateTexture(
        "platform",
        800,
        40
    );


    /* ------------------------------------------------------------
       Lift texture
       ------------------------------------------------------------ */

    const lift = this.make.graphics({
        x: 0,
        y: 0,
        add: false
    });

    lift.fillStyle(0x00ff00, 1);

    lift.fillRect(
        0,
        0,
        100,
        40
    );

    lift.generateTexture(
        "lift",
        100,
        40
    );


    /* ------------------------------------------------------------
       External image assets
       ------------------------------------------------------------ */

    const sittingMelencolo =
        document.getElementById("sittingMelencolo").src;

    this.load.image(
        "sittingMelencolo",
        sittingMelencolo
    );


    const firingMelencolo =
        document.getElementById("firingMelencolo").src;

    this.load.image(
        "firingMelencolo",
        firingMelencolo
    );


    const basketBottom =
        document.getElementById("basketBottom").src;

    this.load.image(
        "basketBottom",
        basketBottom
    );


    const basketRim =
        document.getElementById("basketRim").src;

    this.load.image(
        "basketRim",
        basketRim
    );
}


/* ================================================================
   CREATE
   ================================================================ */

async function create() {

    this.score = 0;


    /* ------------------------------------------------------------
       Score
       ------------------------------------------------------------ */

    this.scoreText = this.add.text(
        20,
        20,
        "Score: 0",
        {
            fontSize: "32px",
            color: "#000",
            fontFamily: "Arial"
        }
    );

    this.scoreText.setDepth(100);


    /* ------------------------------------------------------------
       Platform
       ------------------------------------------------------------ */

    this.platform =
        this.physics.add.staticImage(
            400,
            560,
            "platform"
        );


    /* ------------------------------------------------------------
       Lift
       ------------------------------------------------------------ */

    this.lift =
        this.physics.add.staticImage(
            580,
            520,
            "lift"
        );


    /* ------------------------------------------------------------
       Jars
       ------------------------------------------------------------ */

    this.jars =
        this.physics.add.group();


    /* ------------------------------------------------------------
       Basket
       ------------------------------------------------------------ */

    this.basketBottom =
        this.add.image(
            580,
            500,
            "basketBottom"
        );

    this.basketBottom.setDepth(5);


    this.basketRim =
        this.add.image(
            580,
            500,
            "basketRim"
        );

    this.basketRim.setDepth(15);


    /* ------------------------------------------------------------
       Dragon
       ------------------------------------------------------------ */

    this.dragon =
        this.add.image(
            150,
            450,
            "sittingMelencolo"
        );


    /*
        Make the scene available to app.js.
    */
    window.gameScene = this;
}


/* ================================================================
   START GAME
   ================================================================ */

async function startGame() {

    const scene = window.gameScene;

    console.log("GAME IS STARTING !!");


    if (!scene) {

        console.error(
            "Phaser scene is not ready yet."
        );

        return;
    }


    if (scene.gameTimer) {

        console.log(
            "Game is already running."
        );

        return;
    }


    scene.words = await loadWords();


    /*
        Fire every 4 seconds.
    */
    scene.gameTimer = scene.time.addEvent({

        delay: 4000,

        loop: true,

        callback: () => {

            /*
                Switch to firing frame.
            */
            scene.dragon.setTexture(
                "firingMelencolo"
            );


            spawnJar(scene);


            /*
                After one second go back to sitting.
            */
            scene.time.delayedCall(
                1000,
                () => {

                    scene.dragon.setTexture(
                        "sittingMelencolo"
                    );

                }
            );
        }
    });
}


/* ================================================================
   SPAWN JAR
   ================================================================ */

function spawnJar(scene) {

    const word =
        Phaser.Utils.Array.GetRandom(
            scene.words
        );


    const startX = 180;
    const startY = 500;


    const jar = scene.add.text(
        startX,
        startY,
        "🔥",
        {
            fontSize: "60px",
            fontFamily: "Arial"
        }
    );


    scene.physics.add.existing(jar);

    scene.jars.add(jar);


    jar.body.setBounce(0.1);


    /*
        Phaser's pointer events work with touch input too.
    */
    jar.setInteractive({
        useHandCursor: true
    });


    jar.setDepth(20);


    jar.wordData = word;


    /* ------------------------------------------------------------
       Label
       ------------------------------------------------------------ */

    const label = scene.add.text(
        jar.x - 18,
        jar.y - 10,
        word.de,
        {
            fontSize: "36px",
            color: "#fff"
        }
    );


    jar.label = label;


    /* ------------------------------------------------------------
       Initial velocity
       ------------------------------------------------------------ */

    jar.initialVX =
        jar.body.setVelocityX(
            Phaser.Math.Between(45, 50)
        );


    jar.initialVY =
        jar.body.setVelocityY(-200);


    /* ------------------------------------------------------------
       Touch / click
       ------------------------------------------------------------ */

    jar.on(
        "pointerdown",
        () => showParachute(scene, jar)
    );
}


/* ================================================================
   SHOW PARACHUTE
   ================================================================ */

function showParachute(scene, jar) {

    if (jar.parachuteOpen) {
        return;
    }


    jar.parachuteOpen = true;


    /*
        Pause only this jar.
    */
    jar.body.enable = false;


    const parachute = scene.add.image(
        jar.x,
        jar.y - 70,
        "parachute"
    );


    jar.parachute = parachute;


    const options =
        Phaser.Utils.Array.Shuffle(
            jar.wordData.en
        );


    jar.optionTexts = options.map(
        (opt, i) => {

            const text = scene.add.text(
                jar.x - 60,
                jar.y - 120 + i * 30,
                opt,
                {
                    fontSize: "18px",
                    color: "#000",
                    backgroundColor: "#fff",
                    padding: 5
                }
            ).setInteractive({
                useHandCursor: true
            });


            /*
                Pointerdown works for mouse and touch.
            */
            text.on(
                "pointerdown",
                () => handleChoice(
                    scene,
                    jar,
                    opt
                )
            );


            return text;
        }
    );
}


/* ================================================================
   HANDLE ANSWER
   ================================================================ */

function handleChoice(
    scene,
    jar,
    opt
) {

    let correct = false;


    jar.optionTexts.forEach(
        (t) => {

            t.disableInteractive();


            if (
                opt === jar.wordData.correct &&
                opt === t.text
            ) {

                t.setColor("#2e7d32");

                correct = true;

            }

            else if (
                opt !== jar.wordData.correct &&
                opt === t.text
            ) {

                t.setColor("#c62828");
            }
        }
    );


    scene.time.delayedCall(
        600,
        () => {

            /* ----------------------------------------------------
               Clean up
               ---------------------------------------------------- */

            jar.optionTexts.forEach(
                (t) => t.destroy()
            );


            jar.parachute?.destroy();


            jar.body.enable = true;


            /* ----------------------------------------------------
               Correct answer
               ---------------------------------------------------- */

            jar.setText(
                correct
                    ? "😊"
                    : "💣"
            );


            if (correct) {

                scene.score += 10;

                jar.setDepth(10);


                jar.platformCollider =
                    scene.physics.add.collider(
                        jar,
                        scene.lift,
                        () => {

                            jar.body.setVelocityX(0);

                            jar.body.setGravity(0);
                        }
                    );


                jar.body.setVelocityX(40);
            }


            /* ----------------------------------------------------
               Wrong answer
               ---------------------------------------------------- */

            else {

                scene.score -= 10;


                const tx = 580;
                const ty = 500;


                const x0 = jar.x;
                const y0 = jar.y;


                const dx = tx - x0;
                const dy = ty - y0;


                const len =
                    Math.sqrt(
                        dx * dx +
                        dy * dy
                    );


                const nx = dx / len;
                const ny = dy / len;


                const speed = 400;


                jar.body.setVelocity(
                    nx * speed,
                    ny * speed
                );
            }


            /* ----------------------------------------------------
               Update score
               ---------------------------------------------------- */

            scene.scoreText.setText(
                "Score: " + scene.score
            );
        }
    );
}


/* ================================================================
   UPDATE
   ================================================================ */

function update() {

    this.jars.children.iterate(
        (jar) => {

            if (!jar || !jar.label) {
                return;
            }


            /* ----------------------------------------------------
               Keep label attached to jar
               ---------------------------------------------------- */

            jar.label.x =
                jar.x - 18;

            jar.label.y =
                jar.y - 10;


            /* ----------------------------------------------------
               Keep parachute attached
               ---------------------------------------------------- */

            if (jar.parachute) {

                jar.parachute.x =
                    jar.x;

                jar.parachute.y =
                    jar.y - 70;
            }


            /* ----------------------------------------------------
               Stop jars near bottom
               ---------------------------------------------------- */

            if (
                jar.y > 560 &&
                jar.body.velocity.y < 100
            ) {

                jar.body.setVelocityY(0);

                jar.body.allowGravity = false;
            }


            /* ----------------------------------------------------
               Destroy jars that leave game area
               ---------------------------------------------------- */

            if (jar.y > 620) {

                jar.label.destroy();

                jar.destroy();
            }
        }
    );
}


/* ================================================================
   LOAD WORDS
   ================================================================ */

async function loadWords() {

    const res =
        await fetch("/game_words");


    if (!res.ok) {

        throw new Error(
            `Unable to load game words: HTTP ${ res.status } `
        );
    }


    return await res.json();
}
