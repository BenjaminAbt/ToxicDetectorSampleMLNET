# ML.NET - Toxic Sentiment Sample

This code example shows the use of ML.NET using sentiment detection for toxic or offensive messages.
This can be integrated e.g. in chat applications.

## Test Data
In this sample we use sample data of Kaggle: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

Due to licensing requirements, this sample does not contain that training data.
These must be downloaded yourself and replaced with the train.csv file.

## Usage

After the `train.csv` has been downloaded, the application can simply be started with `dotnet run`.

When started, the application is compiled and the model is trained. In my case, this takes around 5 seconds to generate the model; but can vary depending on processing power (especially CPU power).

```sh
PS C:\source\ba\ToxicDetectorSampleMLNET> dotnet run
=> create model context...
=> loading test data...
=> parsing test data...
=> create model partitions...
=> create binary classification pipeline...
=> model training...
```
When the model was created, it will be validated:
```sh
=> model validation...
=> metrics for binary classification
┌───────────────────────────────────┬─────────┐
│                              Name │ Value   │
├───────────────────────────────────┼─────────┤
│                          Accuracy │ 95,76 % │
│ Area Under Precision Recall Curve │ 84,38 % │
│              Area Under Roc Curve │ 96,42 % │
│                          F1 Score │ 74,15 % │
│                          Log Loss │ 0,18    │
│                Log Loss Reduction │ 0,61    │
│                Positive Precision │ 0,89    │
│                   Positive Recall │ 0,64    │
│                Negative Precision │ 0,96    │
│                   Negative Recall │ 0,99    │
└───────────────────────────────────┴─────────┘
```

Then you can enter your text to check for toxic content:

```sh
--------------------------------------------------
Text Input: Ben likes you

Predict Results:
┌─────────────┬───────────────┐
│        Name │ Value         │
├─────────────┼───────────────┤
│        Text │ Ben likes you │
│  Prediction │ Friendly :-)  │
│ Probability │ 20,68 %       │
│       Score │ -3,3611276    │
└─────────────┴───────────────┘
--------------------------------------------------
```

## Dependencies

### CSVHelper

The ML.NET CSV Reader is very basic and does not support all CSV scenarios. The standard ML NET CSV Reader unfortunately does not work with the CSV files of the Kaggle toxic comment data sample.

Therefore, the CSVHelper (https://github.com/JoshClose/CsvHelper) library is used.

## spectre.console

https://github.com/spectreconsole/spectre.console for nicer console output


## Thank you!

Please donate - if possible - to necessary institutions of your choice such as child cancer aid, children's hospices etc. Thanks!
