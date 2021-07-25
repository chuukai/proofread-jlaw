# proofread-jlaw
日本の法令を校正するための機械学習モデル
## モデルの作り方
### 環境
Google Colab Pro(有料)
### 法令データをダウンロード
[XML一括ダウンロード](https://elaws.e-gov.go.jp/download/)のページで左側のメニューから「法令分類データ」を選び、50種のデータをそれぞれダウンロードし、伸張します。
### XMLファイルから文章を抜き取り
#### 昭和22年以降に制定された法令
XMLファイル(ファイル名の先頭が322から364,401から431,501から503)からSentenceタグの中身を抜き取ります。私の場合はnkfと、xml-twig-toolsパッケージの中にあるxml_grepを使いました。  
コマンドを例示すると
```grepsentence.sh
nkf -w ./37/332AC0000000035_20200401_429AC0000000045/332AC0000000035_20200401_429AC0000000045.xml | xml_grep 'Sentence' --text_only /dev/stdin > ./37/332AC0000000035_20200401_429AC0000000045.sentence
```
となります。
#### 昭和21年以前の法令
文語文になっているため、機械学習の対象としませんでした。
### 強調点の「ヽ」を削除、「つ」を小書きの「っ」に変換
小書きの「っ」は昭和まで使われておらず、このままでは分かち書きに支障が生じるため、「つ」を小書きの「っ」に変換します。  
コマンドを例示すると
``` nomalizeall.sh
cat ./37/332AC0000000035_20200401_429AC0000000045.sentence | sh normalize2.sh > ./37/332AC0000000035_20200401_429AC0000000045.sentence.nml
```
``` normalize2.sh
cat - | perl -p -Mutf8 -CSD -e 's/ヽ//g;' -e 's/であつて/であって/g;' -e 's/であつた/であった/g;' -e 's/のあつた/のあった/g;' -e 's/にあつた/にあった/g;' -e 's/あつては/あっては/g;' -e 's/をもつて/をもって/g;' -e 's/によつて/によって/g;' -e 's/しなかつた/しなかった/g;' -e 's/ねじつた/ねじった/g;' -e 's/をはつた/をはった/g;' -e 's/にそつて/にそって/g;' -e 's/にはつて/にはって/g;' -e 's/つづつた/つづった/g;' -e 's/とつた/とった/g;' -e 's/なつた/なった/g;' -e 's/かかつた/かかった/g;' -e 's/いつて/いって/g;' -e 's/(\p{Han})つ([たて])/$1っ$2/g;'
```
となります。

## ここから先はpublicbunruiwindow.ipynbのソースを見ていただいたほうが早い
1. Mecabとmecab-ipadic-NEologdで個々のファイルを分かち書きしつつ、品詞単位でStopwordを削る。
2. 機械学習モデルを規定文章量ごとに作成するため、未施行などにより重複した法令から古いもの1つにし、ファイルサイズを調べてbunruiwall-sort-uniq2.csvを作成する。
3. 個々のファイルを結合して規定文章量ごとのファイルを作成する。
4. 規定文章量ごとのファイルを対象として、単語をfasttextでベクトル化する。
5. 規定文章量ごとのファイルにある単語ごとに前５単語後ろ５単語をくっつけてデータセットを作る。
6. 規定文章量ごとに30回のmodel.fit後に機械学習モデルを保存する。
