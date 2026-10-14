import yt_dlp

url = ["https://youtu.be/Yz4LlDSlkns?si=2iBkaOqwAZQx1nt4",
        "https://youtu.be/tyB0ztf0DNY?si=70AViI39mXHRR_6V",
        "https://youtu.be/mLfjzJsN8us?si=OaooGyvEW8ZS-U5E",
        "https://youtu.be/EgG3jsGoPvQ?si=cFXhRWnHJ9K3Ty6H",
        "https://youtu.be/Kmh3rhyEtB8?si=6LfaOeJMZvLVem5S",
        "https://youtu.be/GrMBfJNk_NY?si=7pyRzIqu4BfQquyF",
        "https://youtu.be/3WaxQMELSkw?si=l3p3Xlf714iqbyjr",
        "https://youtu.be/AE39gJYuRog?si=XECIwJ8XoTU3dPwV",
        "https://youtu.be/sdE0A2Oxofw?si=pQRz3pfisPjS8lY0",
        "https://youtu.be/TmhpgXScLyY?si=UsE_ZceYq1NGLsPG",
        "https://youtu.be/_rgTlyky1uQ?si=SG4wpiIHlrN8lxhV",
        "https://youtu.be/SrP-PiLSYC0?si=wVEit9XEr5AUnpVl",
        "https://youtu.be/N_aJ5qQbYA0?si=UJ5Z79sVuiTht9Rz",
        "https://youtu.be/QGfn7JeXK54?si=zPyyndtYDBlhjCON",
        "https://youtu.be/fWX9xDmIzRI?si=ZOpI5b4oKHt1_bRR",
        "https://youtu.be/7win3dcgo3k?si=nglXcEp1g7T9hXaG",
        "https://youtu.be/GS_OqZb2CWc?si=LDGDgK2yOOhxUr4w",
        "https://youtu.be/ZHyb-A2Mte4?si=CHb-9LuAaohcpxfp",
        "https://youtu.be/zoilQD1kYSg?si=6k79n0GV-9sX_nVa",
        "https://youtu.be/GqOmJHQZivw?si=VvzTXOefmE_C1xKQ",
        "https://youtu.be/myPeWb3Y68A?si=QEYbBe9nTHMJfx30",
        "https://youtu.be/b3GD8263-PQ?si=vQxndsJ5D3jaQ79D",
        "https://youtu.be/HgyouUi11zk?si=_jFPq8qc-NgEgTxT",
        "https://youtu.be/OgvOZ6OrJoY?si=ZWA4-40OG9C_8hTk",
        "https://youtu.be/mO8XpGoJwuo?si=P5tIpzkJjSY0R-J2",
        "https://youtu.be/NPZn9jBrX8U?si=UIW6AL7qzxG6TmMj",
        "https://youtu.be/-zI4mrF2Pb4?si=83jV0BWOPoVE9vBU",
        "https://youtu.be/_wP9mWNPL5w?si=wBpC-NDQ6PHBQOLp",
        "https://youtu.be/6i_T5kkfv4A?si=38bCPWDoYI_AAjfe",
        "https://youtu.be/xPBLEj41rFU?si=UBZ2bP_h_h3ztBhH",
        "https://youtu.be/yMnH0jrir0Q?si=F9_GsWuluUxDX_Jc"]


ydl_opts = {
    "format": "bestaudio/best",#downloads the best available audio
    "outtmpl": "data/test_audio/%(id)s.%(ext)s"#format of saving the file id->id of the video and ext is the extension of that video
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:#yotubedl is the downloder object into which we give the ydl_opts for setting the configuration
    ydl.download(url)