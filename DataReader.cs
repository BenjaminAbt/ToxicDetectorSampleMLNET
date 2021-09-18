// Copyright © Benjamin Abt 2021, all rights reserved

using CsvHelper;
using CsvHelper.Configuration;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace ToxicSentimentSample
{
    public static class DataReader
    {
        public static List<TrainInput> ReadTrainData(string trainDataFile)
        {
            // we use an external CSV reader because the built in CSV reader of ML.NET
            //   does not support all csv cases. The used train data here is an example for that... :-/

            CsvConfiguration csvConfig = new(CultureInfo.InvariantCulture)
            {
                PrepareHeaderForMatch = args => args.Header.ToLower(),
            };

            using StreamReader reader = new(trainDataFile);
            using CsvReader csv = new(reader, csvConfig);

            return csv.GetRecords<TrainInput>().ToList();
        }
    }
}
