cd embs;
for lang in $(cat ../langs); do
    echo $lang;
    wget https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.${lang}.align.vec;
    wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.${lang}.zip;
    unzip wiki.${lang}.zip;
done
