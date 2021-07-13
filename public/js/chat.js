const socket = io();
const messages = document.querySelector(".chats");
const sendButton = document.querySelector(".send-button");
const input = document.querySelector(".txt-area");
const name = document.querySelector("#name");
const sendFile = document.querySelector("#sendFile");
let myVideo = document.querySelector("#me");
let otherVideo = document.querySelector("#them");
const videoContainer = document.querySelector("#video-container");
const endCallButton = document.querySelector("#endCallButton");
const cameraButton = document.querySelector("#cameraButton");
const micButton = document.querySelector("#micButton");
const readChatButton = document.querySelector("#readChatButton");
const start_button = document.querySelector("#start_button");
const timer_container = document.querySelector("#timer_container");
const s_to_t_button = document.querySelector("#s_to_t_button");
var result;
var startRecording;
let sendVideo = camera;
let sendAudio = mic;
let audioTrack, audioTrack2, videoTrack, videoTrack2;
let readChat = true;
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext = new AudioContext();
const myPeer = new Peer();
myPeer.on("open", function (id) {
  console.log("My peer ID is: " + id);
  socket.emit("userConnected", {
    peerId: id,
    roomId: roomId,
  });
});
let readChatFunction = () => {
  if (readChat) {
    readChatButton.classList.add("btn-info");
    readChatButton.classList.remove("btn-danger");
    readChatButton.innerText = "read Chat";
  } else {
    readChatButton.classList.remove("btn-info");
    readChatButton.classList.add("btn-danger");
    readChatButton.innerText = "mute Chat";
  }
  readChat = !readChat;
};
readChatButton.addEventListener("click", readChatFunction);
//send data test
// myPeer.on("connection", (conn) => {
//   conn.on("data", (data) => {
//     console.log("receiving data");
//     console.log(data);
//   });
// });

socket.on("chatMessage", (message) => {
  if (message.lang == "en" && readChat) {
    responsiveVoice.speak(message.data, "UK English Male");
  } else if (readChat) responsiveVoice.speak(message.data, "Arabic Male");
  outputMessage(
    message.data,
    "left",
    "https://bootdey.com/img/Content/avatar/avatar1.png",
    message.date
  );
});
let now = new Date();
socket.emit("joinRoom", {
  username: `${userFirstName} ${userLastName}`,
  data: `user ${userFirstName} ${userLastName} joined the chat`,
  date: now.toLocaleString(),
  name: user,
  roomId: roomId,
});

sendButton.addEventListener("click", () => {
  let date = new Date();
  socket.emit("chatMessage", {
    data: input.value,
    date: date.toLocaleString(),
    name: user,
    roomId: roomId,
  });
  outputMessage(
    input.value,
    "right",
    "https://bootdey.com/img/Content/avatar/avatar2.png",
    date.toLocaleString()
  );
});

const outputMessage = (message, placement, imageSrc, time) => {
  let chat = document.createElement("div");
  chat.classList.add("chat");
  if (placement === "left") chat.classList.add("chat-left");
  let chatBody = document.createElement("div");
  chatBody.classList.add("chat-body");
  let chatContent = document.createElement("div");
  chatContent.classList.add("chat-content");
  let chatAvatar = document.createElement("div");
  chatAvatar.classList.add("chat-avatar");
  let a = document.createElement("a");
  let i = document.createElement("i");
  let img = document.createElement("img");
  img.src = imageSrc;
  a.className += "avatar avatar-online";
  a.setAttribute("data-toggle", "tooltip");
  a.setAttribute("href", "#");
  a.setAttribute("data-placement", placement);
  a.appendChild(img);
  a.appendChild(i);
  chatAvatar.appendChild(a);
  chat.appendChild(chatAvatar);
  let p = document.createElement("p");
  p.innerText = message;
  let t = document.createElement("time");
  t.dateTime = time;
  t.innerText = time;
  chatContent.appendChild(p);
  chatContent.appendChild(t);
  chatBody.appendChild(chatContent);
  chat.appendChild(chatBody);
  messages.appendChild(chat);
};

// receiving call request and answering it
var rec;
navigator.mediaDevices
  .getUserMedia({ video: true, audio: true })
  .then((stream) => {
    [audioTrack, videoTrack] = stream.getTracks();
    let myStream = stream.clone();
    let audioStream = stream.clone();
    audioStream.removeTrack(audioStream.getTracks()[1]);
    ///initial conditions
    if (mic == false) {
      myStream.getTracks()[0].enabled = false;
      stream.getTracks()[0].enabled = false;
    }
    if (camera == false) {
      myStream.getTracks()[1].enabled = false;
      stream.getTracks()[1].enabled = false;
    }
    myVideo.srcObject = myStream;
    myVideo.srcObject.removeTrack(myVideo.srcObject.getTracks()[0]);

    ////recorder code (for audio)
    const mediaRecorder = new MediaRecorder(audioStream);
    var input = audioContext.createMediaStreamSource(audioStream);
    let audioChunks = [];
    let start_audio_record = true;
    let record_time_out;
    let record_interval;
    startRecording = () => {
      if (start_audio_record) {
        rec = new Recorder(input, {
          numChannels: 1,
        });
        //mediaRecorder.start();
        rec.record();
        audioChunks = [];
        s_to_t_button.classList.replace("btn-info", "btn-danger");
        s_to_t_button.innerText = "stop_record";
        start_timer(20);
        record_time_out = setTimeout(() => {
          //mediaRecorder.stop();
          rec.stop();
          rec.exportWAV(stopRecording);
          s_to_t_button.classList.replace("btn-danger", "btn-info");
          s_to_t_button.innerText = "Speech to text";
        }, 20000);
      } else {
        s_to_t_button.classList.replace("btn-danger", "btn-info");
        s_to_t_button.innerText = "Speech to text";
        clearTimeout(record_time_out);
        clearInterval(record_interval);
        timer_container.innerText = "Not Recording";
        //mediaRecorder.stop();
        rec.stop();
        rec.exportWAV(stopRecording);
      }
      start_audio_record = !start_audio_record;
    };
    s_to_t_button.addEventListener("click", () => {
      if (start_audio_record) {
        rec = new Recorder(input, {
          numChannels: 1,
        });
        //mediaRecorder.start();
        rec.record();
        audioChunks = [];
        s_to_t_button.classList.replace("btn-info", "btn-danger");
        s_to_t_button.innerText = "stop_record";
        start_timer(20);
        record_time_out = setTimeout(() => {
          //mediaRecorder.stop();
          rec.stop();
          rec.exportWAV(stopRecording);
          s_to_t_button.classList.replace("btn-danger", "btn-info");
          s_to_t_button.innerText = "Speech to text";
        }, 20000);
      } else {
        s_to_t_button.classList.replace("btn-danger", "btn-info");
        s_to_t_button.innerText = "Speech to text";
        clearTimeout(record_time_out);
        clearInterval(record_interval);
        timer_container.innerText = "Not Recording";
        //mediaRecorder.stop();
        rec.stop();
        rec.exportWAV(stopRecording);
      }
      start_audio_record = !start_audio_record;
    });

    // mediaRecorder.addEventListener("dataavailable", (event) => {
    //   audioChunks.push(event.data);
    // });
    var stopRecording = (blob) => {
      // if (audioChunks.length != 0) {
      //   console.log(audioChunks);
      //   const audioBlob = new Blob(audioChunks, {
      //     type: "audio/wav",
      //   });
      // console.log(audioBlob);
      // var reader = new FileReader();
      // // reader.readAsArrayBuffer(audioBlob);
      // // reader.onload = () => {
      // //   var result = reader.result;
      // //   // result = result.replace(/^data:audio\/wav;base64,/, "");
      // //   console.log(result);
      // // };
      var xhr = new XMLHttpRequest();
      xhr.withCredentials = true;
      xhr.addEventListener("readystatechange", function () {
        if (this.readyState === 4) {
          result = this.responseText;
          console.log(this.responseText);
          let date = new Date();
          result = JSON.parse(result);
          socket.emit("chatMessage", {
            data: result["DisplayText"],
            date: date.toLocaleString(),
            name: user,
            roomId: roomId,
          });
          outputMessage(
            result["DisplayText"],
            "right",
            "https://bootdey.com/img/Content/avatar/avatar2.png",
            date.toLocaleString()
          );
        }
      });
      // // var fd = new FormData();
      // // fd.append("data-binary", audioBlob, "test.wav");
      xhr.open(
        "POST",
        "https://centralus.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=ar-EG"
      );
      xhr.setRequestHeader("Content-type", "audio/wav");
      xhr.setRequestHeader(
        "Ocp-Apim-Subscription-Key",
        "8c6ed815e5ec4296aa8b060a89874863"
      );

      xhr.send(blob);

      // socket.emit("audio", {
      //   blobData: blob,
      //   roomId: roomId,
      // });
      //}
    };
    //mediaRecorder.addEventListener("stop",stopRecording);

    ////recorder code (for video)
    let recorder = RecordRTC(myStream, {
      MimeType: "video/MKV",
      type: "video",
    });
    function start_timer(i) {
      let time = i;
      let timer = `Record Ends In: 00:${time}`;
      timer_container.innerText = timer;
      let count = () => {
        time--;
        timer = `Record Ends In: 00:${time}`;
        timer_container.innerText = timer;
        console.log(time);
        if (time == 0) {
          clearInterval(record_interval);
          timer = `Not Recording`;
          timer_container.innerText = timer;
        }
      };
      record_interval = setInterval(count, 1000);
    }
    var stop_record = () => {
      recorder.stopRecording(function () {
        let blob = recorder.getBlob();
        start_button.disabled = false;
        // sending the video to the backend
        socket.emit("video", {
          blobData: blob,
          roomId: roomId,
        });
        timer_container.innerText = "Not Recording";
      });
    };
    start_button.addEventListener("click", () => {
      start_button.disabled = true;
      recorder.startRecording();
      start_timer(10);
      setTimeout(stop_record, 10000);
    });
    ///////////////
    myVideo.play();
    cameraButton.addEventListener("click", () => {
      if (sendVideo == true) {
        videoTrack.enabled = false;
        myStream.getTracks()[0].enabled = false;
      } else {
        videoTrack.enabled = true;
        myStream.getTracks()[0].enabled = true;
      }
      sendVideo = !sendVideo;
    });
    micButton.addEventListener("click", () => {
      if (sendAudio == true) {
        audioTrack.enabled = false;
      } else audioTrack.enabled = true;
      sendAudio = !sendAudio;
    });
    myPeer.on("call", (call) => {
      call.answer(stream);
      call.once("stream", (userVideoStream) => {
        [audioTrack2, videoTrack2] = userVideoStream.getTracks();
        otherVideo.srcObject = userVideoStream;
        otherVideo.addEventListener("loadedmetadata", () => {
          otherVideo.play();
        });
      });
      call.once("close", () => {
        videoTrack2.enabled = false;
      });
      endCallButton.addEventListener("click", () => {
        call.close();
        socket.emit("callClosed", { roomId: roomId });
      });
      socket.once("callClosed", () => {
        videoTrack2.enabled = false;
      });
    });

    // sending call request
    socket.on("userConnected", (peerId) => {
      console.log(`calling user ${peerId}`);
      const conn = myPeer.connect(peerId);
      // sendFile.onchange = (event) => {
      //   const file = event.target.files[0];
      //   const blob = new Blob(event.target.files, { type: file.type });
      //   conn.send({
      //     file: blob,
      //     fileName: file.name,
      //     fileType: file.type,
      //   });
      // };
      const call = myPeer.call(peerId, stream);
      call.once("stream", (videoStream) => {
        [audioTrack2, videoTrack2] = videoStream.getTracks();
        ////////setting up video for first connection///////////////////////////////////////////////
        otherVideo.srcObject = videoStream;
        ///////////////////////////////////////////////////////
      });
      call.once("close", () => {
        videoTrack2.enabled = false;
      });
      socket.once("callClosed", () => {
        //otherVideo.parentElement.remove(otherVideo);
        videoTrack2.enabled = false;
        // document.location.href = "../";
      });
      endCallButton.addEventListener("click", () => {
        call.close();
        socket.emit("callClosed", { roomId: roomId });
      });
    });
  });
// .catch(function (err) {
//   console.log("An error occurred: " + err);
// });
