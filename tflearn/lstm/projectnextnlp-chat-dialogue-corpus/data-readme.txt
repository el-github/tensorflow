
対話データファイル書式について
2015年6月11日

1 ファイル書式
1.1 ファイル命名規則
ファイルは1対話1ファイルになっている．ファイル名は「対話のID.log.json」となっている．
対話のIDは後述するdialogue-idと同じである．

1.2 ファイル書式
ファイルの書式を以下に説明する．各対話はシステムから始まり，システム・ユーザが交互に
発話し，システム発話11個，ユーザ発話10個の発話データからなる．エンコード　UTF8，改行
コード　LFのJSONファイルである．JSONの各フィールドの内容について説明する．
・dialogue-id :対話のID
・group-id :WS_PNN23_dial.pdfで説明したが，全対話1146をinit100とrest1046に分割してい
　　　　　　る．init100はgroup-idがinit100になっている．rest1046は全体をランダムにa-k
　　　　　　のサブセットに分けてそれぞれにアノテータを変えてアノテーションを行ってい
　　　　　　るので，subset a～subset kがついている．
・speaker-id :話者のID
・turns :発話データの配列

次に，turnsの中のフィールドの内容について説明する．turnsは発話データの配列になっている．
発話データは以下のフィールドからなる．
・annotations :発話につけられたアノテーション情報（システム発話のみ）
・speaker :話者　ユーザ(U)かシステム(S)
・time :発話入力が完了した時刻
・turn-index :対話内の発話順（0オリジン，数字）
・utterance :発話内容

次に，annotationsの中のフィールドの内容について説明する．このフィールドはアノテータが
発話につけたアノテーションデータが格納されている．各発話について複数人のアノテータが
アノテートを行っているので，配列になっている．
アノテーションデータは以下のフィールドからなる．
・annotator-id :アノテータのID
・breakdown :対話破綻かどうか(O/T/X)
　O：破綻ではない　○
　T：破綻とは言い切れないが，違和感を感じる発話　△
　X：あきらかにおかしいと思う発話，破綻　×
・comment :アノテータがつけたコメント
・ungrammatical-sentence :非文かどうか
　O：非文ではない
　X：非文である
＊turn-index以外は全て文字列フィールドである．

1.3 話者
話者はProject Next NLP　対話タスクに参加している15拠点から集められている．各話者のID
は「拠点ID_拠点内話者ID」となっており，拠点IDは01～15，拠点内話者IDは01から順番につけ
られている．別ファイル(speaker-attribute.tsv)に話者の性別，年代，職業の属性情報を示す．

1.4 アノテータ
アノテータはProject Next NLP　対話タスクに参加している15拠点から集められている．各ア
ノテータのIDは「拠点ID_拠点内アノテータID」となっており，拠点IDは01～15，拠点内アノテ
ータIDはAから順番につけられている．拠点IDは話者IDについているものと共通なので，同じ
IDは同じ拠点を意味する．ただし，拠点内話者IDと拠点内アノテータIDは対応していないので，
01_01と01_Aが同じ人物とは限らない．
アノテーションは2回に分けて実施されている．詳細はWS_PNN23_dial.pdfに記載されている．
1回目は1,146対話からランダムに100対話を抽出し，これをinit100とした．この全データに対
して24名でアノテーションを行った．2回目は残りの1046対話(rest1046)に対して22名でアノテ
ーションを行った．このうちの19名は1回目のアノテーションに参加したアノテータである．こ
の19名は1回目のアノテーション結果から2つのクラスタに分類されている．まずこの19名につ
いて，2つのクラスタからなるべく1名づつのアノテータが割り当てられるように，サブセット k
を除く10サブセットに割り当てた．その後残りの3名を同10サブセットに割り当てた．1名当り
の分担量を2サブセットと固定して22名を10サブセットに割り当てたので，i,jの2つのサブセット
だけ3名のアノテータを割り当てた．サブセット kについては，余力のある2名に割り当てた．
別ファイル(annotator-attribute.tsv)にアノテータの性別，年代，職業の属性情報を示す．
また，init100に参加していたかどうか，どのクラスタに分類されたか(init100時のクラスタ)，
WS_PNN23_dial.pdfの図3の番号に対応する番号(init100時のID)が記載されている．

＊話者とアノテータは共通の拠点IDを使っている．拠点ID 06は対話データ収集後アノテーション
からの参加のため，拠点ID 06を持つ話者はいない．

1.5注意事項
ユーザ発話・コメントについては自由入力であったため，環境依存文字が含まれていた箇所があり，
変換可能なものは変換をしている．ただし，下記は変換できる文字がなかったため，削除している．
dialogue-id:1408001850 turn-index:3
「なんでもないよ♡」→「なんでもないよ」
dialogue-id:1408002669 turn-index:7
「嘘♡好き♡」→「嘘好き」

改訂履歴
2015.06.11　作成



