using Amazon;
using Amazon.Rekognition;
using Amazon.Rekognition.Model;
using Newtonsoft.Json.Linq;
using Org.OpenAPITools.Api;
using Org.OpenAPITools.Model;

class Program
{
    enum TestType
    {
        AWS, OPEN_CV, DEEP_FAKE_TEST, MODEL_VALIDATION
    }

    private static readonly AmazonRekognitionClient rekognitionClient = new AmazonRekognitionClient("access-key", "secret-key", RegionEndpoint.EUCentral1);
    private static readonly double THRESHOLD_OPEN_CV = 0.7;
    private static readonly double THRESHOLD_AWS = 80;

    static void Main(string[] args)
    {
        int testId = 1;

        //start test by test type and variant
        Test(testId, TestType.DEEP_FAKE_TEST, testWithOriginals: false);
    }

    static void Test(int testId, TestType testType, bool testWithOriginals, DetectorBackend? detectorBackend = null)
    {
        Console.WriteLine("Enter root folder path of test data:");
        string rootFolder = Console.ReadLine();
        Console.WriteLine("Enter path to csv output file");
        string outputCsv = Console.ReadLine();

        //init csv
        using (StreamWriter writer = new StreamWriter(outputCsv))
        {
            writer.AutoFlush = true;

            //number of test images for each variant test
            //for final deep fake test only 2 because of too high costs for LLM
            int numberOfTestImages = 0;

            //helper list for temporary saving already tested original images (only for online test variant)
            HashSet<string> testedOriginalImages = [];

            //csv headers
            var csvHeaderString = $"Test_ID;Original_File_ID;Test_File_ID;Person_Name;Gender;Test_Variant";

            switch(testType)
            {
                case TestType.AWS:
                    numberOfTestImages = 5;
                    writer.WriteLine(string.Join(";", csvHeaderString, "Similarity_AWS", "Faces_Matching_AWS"));
                    break;
                case TestType.OPEN_CV:
                    numberOfTestImages = 5;
                    writer.WriteLine(string.Join(";", csvHeaderString, "Similarity_OpenCV", "Faces_Matching_OpenCV"));               
                    break;
                case TestType.DEEP_FAKE_TEST:
                    numberOfTestImages = 1;
                    writer.WriteLine(string.Join(";", csvHeaderString, "Verified", "Similarity", "Confidence_Score", "Explanation", "Test_Valid"));                                
                    break;
                case TestType.MODEL_VALIDATION:
                    numberOfTestImages = 1;
                    writer.WriteLine("Test_ID;Test_Variant;VGG-Face;Facenet;Facenet512;OpenFace;DeepID;ArcFace;Dlib;SFace;GhostFaceNet");                                
                    break;
            }

            var personFolders = Directory.GetDirectories(rootFolder).OrderBy(folder => ExtractNumericId(folder)).ToList();
            foreach (string personFolder in personFolders)
            {
                string personId = Path.GetFileName(personFolder);
                //load only a few variant folders when doing final deep fake test
                List<string> variantFolders = new List<string>();
                if (testType == TestType.DEEP_FAKE_TEST)
                {
                    variantFolders = Directory.GetDirectories(personFolder).Take(4).OrderBy(x => ExtractNumericId(x)).ToList();
                }
                else
                {
                    variantFolders = Directory.GetDirectories(personFolder).OrderBy(x => ExtractNumericId(x)).ToList();
                }

                foreach(string currentVariantFolder in variantFolders)
                {
                    var imagesUnordered = Directory.GetFiles(currentVariantFolder, "*.jpg");
                    if (imagesUnordered.Length == 0)
                    {
                        continue;
                    }

                    //image with only one underscore is original
                    var originalImage = imagesUnordered.FirstOrDefault(str => str.Count(c => c == '_') == 1);

                    var nameAndGender = GetCelebNameAndGender(originalImage);
                    var name = nameAndGender.Item1;
                    var gender = nameAndGender.Item2;

                    var images = imagesUnordered.Where(x => x != originalImage).ToList();

                    if (images.Count > 0)
                    {
                        // DF Variant 1: DeepFakes in same subfolder
                        var imagesToTest = images.Take(numberOfTestImages).ToList();
                            
                        foreach(string deepFakeImage in imagesToTest)
                        {
                            if (!testWithOriginals)
                            {
                                Console.WriteLine($"Run test ID: {testId}");
                                var variant1ResultEntry = $"{testId};{Path.GetFileName(originalImage)};{Path.GetFileName(deepFakeImage)};{name};{gender}";
                                RunServicesByTestType(testType, testId, 1, writer, variant1ResultEntry, originalImage, deepFakeImage, detectorBackend);    
                                writer.Flush();
                                testId++;
                            }
                        }

                        // DF Variant 2: DeepFakes from another subfolder
                        foreach(string otherVariantFolder in variantFolders)
                        {
                            if (otherVariantFolder == currentVariantFolder) continue; // skip same variant folder

                            string otherVariantId = Path.GetFileName(otherVariantFolder);
                            string[] otherVariantImagesUnordered = Directory.GetFiles(otherVariantFolder, "*.jpg");

                            var originalImageFromOtherVariantFolder = otherVariantImagesUnordered.FirstOrDefault(str => str.Count(c => c == '_') == 1);
                            if (testWithOriginals && !string.IsNullOrEmpty(originalImageFromOtherVariantFolder))
                            {
                                //check if "originalImageFromOtherVariantFolder" was already tested as originalImage to avoid duplicates. when yes, skip this test and continue
                                var imageAlreadyTested = testedOriginalImages.Contains(originalImageFromOtherVariantFolder);
                                if (!imageAlreadyTested)
                                {
                                    //test with original images when needed
                                    Console.WriteLine($"Run test ID: {testId}");
                                    var originalTestResultEntry = $"{testId};{Path.GetFileName(originalImage)};{Path.GetFileName(originalImageFromOtherVariantFolder)};{name};{gender}";
                                    RunServicesByTestType(testType, testId, 0, writer, originalTestResultEntry, originalImage, originalImageFromOtherVariantFolder, detectorBackend);
                                    testedOriginalImages.Add(originalImage);                            
                                    testId++;
                                }
                            }
                            else
                            {
                                //skip original image from another variant
                                otherVariantImagesUnordered = otherVariantImagesUnordered.Where(x => x != originalImageFromOtherVariantFolder).ToArray();
                                var otherVariantImagesToTest = otherVariantImagesUnordered.Take(numberOfTestImages).ToList(); 

                                foreach(string deepFakeImage in otherVariantImagesToTest)
                                {
                                    Console.WriteLine($"Run test ID: {testId}");
                                    var variant2ResultEntry = $"{testId};{Path.GetFileName(originalImage)};{Path.GetFileName(deepFakeImage)};{name};{gender}";
                                    RunServicesByTestType(testType, testId, 2, writer, variant2ResultEntry, originalImage, deepFakeImage, detectorBackend);
                                    writer.Flush();
                                    testId++;
                                }
                            }
                        }
                    }
                }
            }
        }

        Console.WriteLine($"Test results saved in {outputCsv}.");
    }

    static void RunServicesByTestType(TestType testType, int testId, int testVariant,  StreamWriter writer, string resultEntryString, string sourceImagePath, string targetImagePath, DetectorBackend? detectorBackend = null)
    {
        if (testType == TestType.AWS)
        {
            var similarityScoreAWS = GetFaceComparisonSimilarityByAWS(sourceImagePath, targetImagePath);
            var facesMatchingAWS = similarityScoreAWS >= THRESHOLD_AWS;
            writer.WriteLine(string.Join(";", resultEntryString, testVariant, similarityScoreAWS, facesMatchingAWS));
        } 
        else if (testType == TestType.OPEN_CV)
        {    
            var similarityScoreOpenCV = GetFaceComparisonSimilarityByOpenCV(sourceImagePath, targetImagePath);
            var facesMatchingOpenCV = similarityScoreOpenCV >= THRESHOLD_OPEN_CV;
            writer.WriteLine(string.Join(";", resultEntryString, testVariant, similarityScoreOpenCV, facesMatchingOpenCV));
        } 
        else if (testType == TestType.DEEP_FAKE_TEST)
        {
            var deepfakeCheckResult = GetDeepFakeCheckResult(sourceImagePath, targetImagePath);
            //sometimes the llm adds ";" as character to explanation -> replace with " - " for not destroyying csv 
            var explanation = deepfakeCheckResult.Explanation.Replace(";", " -");
            writer.WriteLine(string.Join(";", resultEntryString, testVariant, deepfakeCheckResult.Verified, deepfakeCheckResult.Similarity, deepfakeCheckResult.ConfidenceScore, explanation, deepfakeCheckResult.TestValid));
        }
        else if (testType == TestType.MODEL_VALIDATION)
        {
            var modelValidationResult = GetModelValidationResult((DetectorBackend)detectorBackend, testId, testVariant, sourceImagePath, targetImagePath);
            writer.WriteLine(modelValidationResult);
        }                                          
    }

    static int ExtractNumericId(string folderPath)
    {
        string folderName = Path.GetFileName(folderPath);
        return int.TryParse(new string(folderName.Where(char.IsDigit).ToArray()), out int id) ? id : 0;
    }

    #region services
    private static Tuple<string, string> GetCelebNameAndGender(string sourceImagePath)
    {        
        // Convert image to byte array and create request
        byte[] sourceImageBytes = File.ReadAllBytes(sourceImagePath);
        var sourceImage = new Image { Bytes = new MemoryStream(sourceImageBytes) };
        RecognizeCelebritiesRequest recognizeCelebritiesRequest = new()
        {
            Image = sourceImage
        };

        try
        {
            Console.WriteLine("Run celebrity recognition for name and gender");
            var recognizeCelebResult = rekognitionClient.RecognizeCelebritiesAsync(recognizeCelebritiesRequest).Result;

            var recognizedCeleb = recognizeCelebResult.CelebrityFaces.FirstOrDefault();
            if (recognizedCeleb?.MatchConfidence > 80)
            {
                var recognizedCelebName = recognizedCeleb?.Name;
                var recognizedCelebGender = recognizedCeleb?.KnownGender.Type.Value;

                Console.WriteLine($"celebrity recognition for {recognizedCelebName} finished");
                return new Tuple<string, string>(recognizedCelebName, recognizedCelebGender);
            }
        }
        catch(Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
        
        return new Tuple<string, string>("","");
    }

    private static double GetFaceComparisonSimilarityByAWS(string sourceImagePath, string targetImagePath)
    {
        // Convert images to byte arrays
        byte[] sourceImageBytes = File.ReadAllBytes(sourceImagePath);
        byte[] targetImageBytes = File.ReadAllBytes(targetImagePath);

        // Create requests
        var sourceImage = new Image { Bytes = new MemoryStream(sourceImageBytes) };
        var targetImage = new Image { Bytes = new MemoryStream(targetImageBytes) };

        var compareFacesRequest = new CompareFacesRequest
        {
            SourceImage = sourceImage,
            TargetImage = targetImage,
            SimilarityThreshold = 0
        };

        try
        {
            Console.WriteLine($"Comparing faces {sourceImagePath} and {targetImagePath}");
            var response = rekognitionClient.CompareFacesAsync(compareFacesRequest).Result;

            if (response.FaceMatches.Count > 0)
            {
                var firstMatch = response.FaceMatches.FirstOrDefault();
                return firstMatch.Similarity;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
        
        return 0;
    }

    private static double GetFaceComparisonSimilarityByOpenCV(string sourceImagePath, string targetImagePath)
    {
        try
        {
            // Convert images to byte arrays
            byte[] sourceImageBytes = File.ReadAllBytes(sourceImagePath);
            byte[] targetImageBytes = File.ReadAllBytes(targetImagePath);

            // convert byte arrays to base64 strings
            string sourceBase64String = Convert.ToBase64String(sourceImageBytes);
            string targetBase64String = Convert.ToBase64String(targetImageBytes);

            string body = $@"
            {{
            ""gallery"": [
                ""{sourceBase64String}""
            ],
            ""probe"": [
                ""{targetBase64String}""
            ],
            ""search_mode"": ""ACCURATE""
            }}";

            var client = new HttpClient();
            var request = new HttpRequestMessage(HttpMethod.Post, "https://eu.opencv.fr/compare");
            request.Headers.Add("accept", "application/json");
            request.Headers.Add("x-api-key", "api-key");
            var content = new StringContent(body, null, "application/json");
            request.Content = content;
            var response = client.Send(request);
            response.EnsureSuccessStatusCode();
            var resultJson = JObject.Parse(response.Content.ReadAsStringAsync().Result);
            var scoreString = resultJson.GetValue("score")?.ToString();
            var score = double.Parse(scoreString);
            return score;
        }
        catch(Exception e)
        {
            Console.WriteLine(e);
            return 0;
        }
    }

    private static LLMResultModel GetDeepFakeCheckResult(string sourceImagePath, string targetImagePath)
    {
        var defaultApi = new DefaultApi("http://127.0.0.1:8000/");
        try
        {
            FileStream originalFileStream = new FileStream(sourceImagePath, FileMode.Open);
            FileStream secondFileStream = new FileStream(targetImagePath, FileMode.Open);
            var faceDetectionModel = defaultApi.CheckFacesFaceDetectionPost(originalFileStream, secondFileStream);
            var faceDetectionModelJson = faceDetectionModel.ToJson();
            originalFileStream.Dispose();
            secondFileStream.Dispose();

            FileStream originalFileStreamLLM = new FileStream(sourceImagePath, FileMode.Open);
            FileStream secondFileStreamLLM = new FileStream(targetImagePath, FileMode.Open);
            var llmResult = defaultApi.VerifyByLmmCheckByLLMPost(faceDetectionModelJson, originalFileStreamLLM, secondFileStreamLLM);
            originalFileStreamLLM.Dispose();
            secondFileStreamLLM.Dispose();
            return llmResult;
        }
        catch (Exception e)
        {
            return new LLMResultModel(false, 0, 0, "Error", false);
        }
    }

    private static string GetModelValidationResult(DetectorBackend detectorBackend, int testId, int testVariant, string sourceImagePath, string targetImagePath)
    {
        var models = new List<string>{
            "VGG-Face", 
            "Facenet", 
            "Facenet512", 
            "OpenFace", 
            "DeepID", 
            "ArcFace", 
            "Dlib", 
            "SFace",
            "GhostFaceNet"
        };

        string resultCSVEntry = $"{testId};{testVariant}";

        var defaultApi = new DefaultApi("http://127.0.0.1:8000/");

        try
        {
            FileStream originalFileStream = new FileStream(sourceImagePath, FileMode.Open);
            FileStream secondFileStream = new FileStream(targetImagePath, FileMode.Open);

            List<DeepFaceVerificationModel> validationResults = defaultApi.ValidateModelsModelValidationPost(detectorBackend, originalFileStream, secondFileStream);
            for (int i = 0; i < validationResults.Count; i++)
            {
                var usedModel = models[i];
                var modelResult = validationResults[i];
                if (usedModel == modelResult.Model)
                {
                    bool testVerified = false;
                    var deepfaceVerify = modelResult.DeepfaceVerify;

                    //check if deep face test is verified
                    //When Test Variant is 0 (original images test) check if DeepFace model verified if these are from same person
                    //When Test Variant is 1 or 2 (deep fake tests) check if DeepFace model verified if these are from different persons (invert verify result)
                    if (testVariant == 0)
                    {
                        testVerified = modelResult.DeepfaceVerify;
                    }
                    else if (testVariant == 1 || testVariant == 2)
                    {
                        testVerified = !modelResult.DeepfaceVerify;
                    }
                    resultCSVEntry = string.Join(";", resultCSVEntry, testVerified);
                }
                else
                {
                    resultCSVEntry = string.Join(";", resultCSVEntry, "");
                }
            }   

            return resultCSVEntry;
        }
        catch
        {
            return resultCSVEntry;
        }
    }
    #endregion
}