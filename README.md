# proofread-jlaw
日本の法令を校正するための機械学習モデル



## モデルの作り方
### 環境
Google Colab Pro(有料)
### 法令データのダウンロード
「XML一括ダウンロード」のページ https://elaws.e-gov.go.jp/download/ で左側のメニューから「法令分類データ」を選び、50種のデータをそれぞれダウンロードします。
### 文章の抜き取り
XMLファイルからSentenceタグの中身を抜き取ります。私の場合はnkfと、xml-twig-toolsパッケージの中にあるxml_grepを使いました。
コマンドを例示すると
nkf -w ./37/332AC0000000035_20200401_429AC0000000045/332AC0000000035_20200401_429AC0000000045.xml | xml_grep 'Sentence' --text_only /dev/stdin > ./37/332AC0000000035_20200401_429AC0000000045.sentence
となります。
