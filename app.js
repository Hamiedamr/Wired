const express = require("express"),
  { spawn } = require("child_process"),
  fs = require("fs"),
  FileReader = require("filereader"),
  http = require("http"),
  app = express(),
  socketio = require("socket.io"),
  server = http.createServer(app),
  io = socketio(server),
  User = require("./models/User"),
  flash = require("connect-flash"),
  mongoose = require("mongoose"),
  url = require("url"),
  bodyparser = require("body-parser"),
  passport = require("passport"),
  uuidv4 = require("uuid").v4,
  nodemailer = require("nodemailer"),
  crypto = require("crypto"),
  async = require("async"),
  localStrategy = require("passport-local");
mongoose.connect(
  process.env.MONGODB_URI ||
    "mongodb+srv://aukshmark:aukshmark15@cluster0.nliro.mongodb.net/PHASE1?retryWrites=true&w=majority",
  {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  }
);
/////////////////////////////////
/////////////////////////////////
let port = process.env.PORT || 3000;
app.set("view engine", "ejs");
app.use(
  require("express-session")({
    secret: "mark",
    saveUninitialized: false,
    resave: false,
  })
);
app.use(flash());
app.use(express.static("public"));
app.use(express.static("node_modules"));
app.use(bodyparser.urlencoded({ extended: true }));
app.use(passport.initialize());
app.use(passport.session());
passport.use(new localStrategy(User.authenticate()));
passport.serializeUser(User.serializeUser());
passport.deserializeUser(User.deserializeUser());

let MAIL = "WIRED.APP.2021@gmail.com",
  MAILPW = process.env.Password;
/////////////////////////////////////////////////////////
//routes/////////////////////////////////////////////////
/////////////////////////////////////////////////////////
function isLogged(req, res, next) {
  if (req.isAuthenticated()) return next();
  req.flash("error", "please login first");
  res.redirect("/login");
}
io.on("connection", (socket) => {
  socket.on("joinRoom", (data) => {
    socket.join(data.roomId);
    // data.lang = "en";
    socket.broadcast.to(data.roomId).emit("chatMessage", data);
  });
  socket.on("chatMessage", (text) => {
    console.log(text);
    var msg = text;
    socket.to(text.roomId).emit("chatMessage", msg);
  });
  socket.on("img", (data) => {
    var base64Data = data.data.replace(/^data:image\/png;base64,/, "");
    var roomId = data.roomId;
    //A Blob() is almost a File() - it's just missing the two properties below which we will add
    img_name = "test.jpg";
    fs.writeFile("images/" + img_name, base64Data, "base64", function (err) {
      // console.log(err);
      return;
    });
    console.log("image saved");
    const python = spawn("python", [
      "GP_Blind_Features/blind_features.py",
      data.type,
    ]);
    python.stdout.on("data", (out) => {
      dataFromPython = out.toString();
      // console.log(dataFromPython)
      // console.log(data.type)
      io.to(socket.id).emit(data.type, dataFromPython);
    });
  });
  socket.on("video", (data) => {
    let blob = data.blobData;
    var roomId = data.roomId;
    //A Blob() is almost a File() - it's just missing the two properties below which we will add
    blob.lastModifiedDate = new Date();
    blob.name = "test.MKV";
    // console.log(blob);
    var reader = new FileReader();
    reader.onload = function () {
      var buffer = new Buffer(reader.result);
      fs.writeFile(`videos/${blob.name}`, buffer, {}, (err, res) => {
        if (err) {
          // console.error(err);
          return;
        }
        console.log("video saved");
        const python = spawn("python", ["GP_Deaf_Features/deaf_features.py"]);
        python.stdout.on("data", (data) => {
          dataFromPython = data.toString();
          let now = new Date();
          io.in(roomId).emit("chatMessage", {
            data: dataFromPython,
            date: now.toLocaleString(),
            lang: "en",
            id: socket.id,
          });
        });
      });
    };
    reader.readAsArrayBuffer(blob);
  });
  socket.on("userConnected", (data) => {
    socket.broadcast.to(data.roomId).emit("userConnected", data.peerId);
  });

  socket.on("callClosed", (data) => {
    socket.broadcast.to(data.roomId).emit("callClosed");
  });
});

/////////////////Get requests////////////////////////////

app.get("/login", (req, res) => {
  if (req.isAuthenticated()) res.redirect("/home");
  else res.render("index", { message: req.flash("error") });
});

app.get("/home", isLogged, (req, res) => {
  res.render("home", { user: req.user });
});

app.post("/home/chat", isLogged, (req, res) => {
  console.log(req.body);
  res.redirect(
    url.format({
      pathname: `/home/chat/${uuidv4()}`,
      query: {
        options: req.body.options,
        problems: req.body.problems,
      },
    })
  );
});

app.post("/home/join", isLogged, (req, res) => {
  let link = req.body.chaturl;
  let start = link.search("/chat/") + 6;
  let end = link.search("options") - 1;
  link = link.slice(start, end);
  res.redirect(
    url.format({
      pathname: `/home/chat/${link}`,
      query: {
        options: req.body.options,
        problems: req.body.problems,
      },
    })
  );
});

app.get("/forgot", (req, res) => {
  res.render("forgot");
});

app.get("/story", (req, res) => {
  res.render("story");
});

app.get("/reset/:token", (req, res) => {
  User.findOne(
    {
      resetPasswordToken: req.params.token,
      resetPasswordExpires: { $gt: Date.now() },
    },
    function (err, user) {
      if (!user) {
        // req.flash("error", "Password reset token is invalid or has expired.");
        return res.redirect("/forgot");
      }
      res.render("reset", { token: req.params.token });
    }
  );
  // res.render("reset", { user: req.token });
});

app.post("/forgot", (req, res, next) => {
  async.waterfall(
    [
      function (done) {
        crypto.randomBytes(20, function (err, buf) {
          var token = buf.toString("hex");
          done(err, token);
        });
      },
      function (token, done) {
        User.findOne({ username: req.body.email }, function (err, user) {
          if (!user) {
            // req.flash("error", "No account with that email address exists.");
            console.log("not found");
            return res.redirect("/forgot");
          }

          user.resetPasswordToken = token;
          user.resetPasswordExpires = Date.now() + 3600000; // 1 hour

          user.save(function (err) {
            done(err, token, user);
          });
        });
      },
      function (token, user, done) {
        var smtpTransport = nodemailer.createTransport({
          service: "Gmail",
          auth: {
            user: MAIL,
            pass: MAILPW,
          },
        });
        var mailOptions = {
          to: user.username,
          from: "WIRED",
          subject: "WIRED Password Reset",
          text:
            "You are receiving this because you (or someone else) have requested the reset of the password for your account.\n\n" +
            "Please click on the following link, or paste this into your browser to complete the process:\n\n" +
            "http://" +
            req.headers.host +
            "/reset/" +
            token +
            "\n\n" +
            "If you did not request this, please ignore this email and your password will remain unchanged.\n",
        };
        smtpTransport.sendMail(mailOptions, function (err) {
          console.log("mail sent");
          // req.flash(
          //   "success",
          //   "An e-mail has been sent to " +
          //     user.username +
          //     " with further instructions."
          // );
          done(err, "done");
        });
      },
    ],
    function (err) {
      if (err) return next(err);
      res.redirect("/forgot");
    }
  );
});

app.post("/reset/:token", (req, res) => {
  async.waterfall(
    [
      function (done) {
        User.findOne(
          {
            resetPasswordToken: req.params.token,
            resetPasswordExpires: { $gt: Date.now() },
          },
          function (err, user) {
            if (!user) {
              // req.flash(
              //   "error",
              //   "Password reset token is invalid or has expired."
              // );
              return res.redirect("back");
            }
            if (req.body.password === req.body.confirm) {
              user.setPassword(req.body.password, function (err) {
                user.resetPasswordToken = undefined;
                user.resetPasswordExpires = undefined;

                user.save(function (err) {
                  req.logIn(user, function (err) {
                    done(err, user);
                  });
                });
              });
            } else {
              req.flash("error", "Passwords do not match.");
              return res.redirect("back");
            }
          }
        );
      },
      function (user, done) {
        var smtpTransport = nodemailer.createTransport({
          service: "Gmail",
          auth: {
            user: MAIL,
            pass: MAILPW,
          },
        });
        var mailOptions = {
          to: user.username,
          from: MAIL,
          subject: "Your password has been changed",
          text:
            "Hello,\n\n" +
            "This is a confirmation that the password for your account " +
            user.username +
            " has just been changed.\n",
        };
        smtpTransport.sendMail(mailOptions, function (err) {
          // req.flash("success", "Success! Your password has been changed.");
          done(err);
        });
      },
    ],
    function (err) {
      res.redirect("/login");
    }
  );
});

app.get("/home/chat/:id", isLogged, (req, res) => {
  let camera = false,
    mic = false,
    problems = req.query.problems;
  if (req.query.options.includes("camera") || req.query.options == "camera") {
    camera = true;
  }
  if (
    req.query.options.includes("microphone") ||
    req.query.options == "microphone"
  ) {
    mic = true;
  }
  res.render("chat", {
    user: req.user,
    roomId: req.params.id,
    mic: mic,
    camera: camera,
    problems: problems,
  });
});

app.get("/logout", (req, res) => {
  req.logout();
  res.redirect("/login");
});

/////////////////Post requests////////////////////////////

app.post(
  "/login",
  passport.authenticate("local", {
    successRedirect: "/home",
    failureRedirect: "/login",
    failureFlash: "Invalid username or password.",
  }),
  (req, res) => {}
);

app.post(
  "/sign_up",
  (req, res, next) => {
    if (req.body.password === req.body.passwordRe) {
      return next();
    }
    req.flash("error", "passwords do not match.");
    res.redirect("/login");
  },
  (req, res) => {
    User.register(
      new User({
        username: req.body.username,
      }),
      req.body.password,
      (err, user) => {
        if (err) {
          console.log(err);
          req.flash("error", "email already exists.");
          return res.render("index", { message: req.flash("error") });
        }
        user.firstName = req.body.firstName;
        user.lastName = req.body.lastName;
        user.save();
        console.log(user);
        passport.authenticate("local")(req, res, function () {
          res.redirect("/home");
        });
      }
    );
  }
);

//////////////////////////////////////////////////////////
app.get("*", (req, res) => {
  res.redirect("/login");
});
server.listen(port, () => {
  console.log("listening on http://localhost:3000/login");
});
