wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=17sZxLu9wdbHKBg6bn8-AAqhevhCTF16S' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=17sZxLu9wdbHKBg6bn8-AAqhevhCTF16S" -O ao_only.zip && rm -rf /tmp/cookies.txt

unzip ao_only.zip

rm -rf __MACOSX
mkdir model
mkdir ../data
mkdir results
mv unet_data/UNetModel_set1000_1.pth ./model/
mv unet_data/VidSepModel_set1000_1.pth ./model/
mv unet_data/translator.mp4 ../data/
rm -rf unet_data
rm ao_only.zip

