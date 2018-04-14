echo 'Easy'
python3 classify.py --mode train --algorithm adaboost --model-file trained/easy.adaboost.model --data datasets/easy.train --num-boosting-iterations 10
python3 classify.py --mode test --model-file trained/easy.adaboost.model --data datasets/easy.train --predictions-file predictions/easy.train.predictions
python3 classify.py --mode test --model-file trained/easy.adaboost.model --data datasets/easy.dev --predictions-file predictions/easy.dev.predictions
python3 compute_accuracy.py datasets/easy.train predictions/easy.train.predictions
python3 compute_accuracy.py datasets/easy.dev predictions/easy.dev.predictions
echo 'Hard'
python3 classify.py --mode train --algorithm adaboost --model-file trained/hard.adaboost.model --data datasets/hard.train --num-boosting-iterations 10
python3 classify.py --mode test --model-file trained/hard.adaboost.model --data datasets/hard.train --predictions-file predictions/hard.train.predictions
python3 classify.py --mode test --model-file trained/hard.adaboost.model --data datasets/hard.dev --predictions-file predictions/hard.dev.predictions
python3 compute_accuracy.py datasets/hard.train predictions/hard.train.predictions
python3 compute_accuracy.py datasets/hard.dev predictions/hard.dev.predictions
echo 'Finance'
python3 classify.py --mode train --algorithm adaboost --model-file trained/finance.adaboost.model --data datasets/finance.train --num-boosting-iterations 10
python3 classify.py --mode test --model-file trained/finance.adaboost.model --data datasets/finance.train --predictions-file predictions/finance.train.predictions
python3 classify.py --mode test --model-file trained/finance.adaboost.model --data datasets/finance.dev --predictions-file predictions/finance.dev.predictions
python3 classify.py --mode test --model-file trained/finance.adaboost.model --data datasets/finance.test --predictions-file predictions/finance.test.predictions
python3 compute_accuracy.py datasets/finance.train predictions/finance.train.predictions
python3 compute_accuracy.py datasets/finance.dev predictions/finance.dev.predictions
python3 compute_accuracy.py datasets/finance.test predictions/finance.test.predictions
echo 'Speech'
python3 classify.py --mode train --algorithm adaboost --model-file trained/speech.adaboost.model --data datasets/speech.train --num-boosting-iterations 10
python3 classify.py --mode test --model-file trained/speech.adaboost.model --data datasets/speech.train --predictions-file predictions/speech.train.predictions
python3 classify.py --mode test --model-file trained/speech.adaboost.model --data datasets/speech.dev --predictions-file predictions/speech.dev.predictions
python3 classify.py --mode test --model-file trained/speech.adaboost.model --data datasets/speech.test --predictions-file predictions/speech.test.predictions
python3 compute_accuracy.py datasets/speech.train predictions/speech.train.predictions
python3 compute_accuracy.py datasets/speech.dev predictions/speech.dev.predictions
python3 compute_accuracy.py datasets/speech.test predictions/speech.test.predictions
echo 'Vision'
python3 classify.py --mode train --algorithm adaboost --model-file trained/vision.adaboost.model --data datasets/vision.train --num-boosting-iterations 10
python3 classify.py --mode test --model-file trained/vision.adaboost.model --data datasets/vision.train --predictions-file predictions/vision.train.predictions
python3 classify.py --mode test --model-file trained/vision.adaboost.model --data datasets/vision.dev --predictions-file predictions/vision.dev.predictions
python3 classify.py --mode test --model-file trained/vision.adaboost.model --data datasets/vision.test --predictions-file predictions/vision.test.predictions
python3 compute_accuracy.py datasets/vision.train predictions/vision.train.predictions
python3 compute_accuracy.py datasets/vision.dev predictions/vision.dev.predictions
python3 compute_accuracy.py datasets/vision.test predictions/vision.test.predictions
echo 'Bio'
python3 classify.py --mode train --algorithm adaboost --model-file trained/bio.adaboost.model --data datasets/bio.train --num-boosting-iterations 10
python3 classify.py --mode test --model-file trained/bio.adaboost.model --data datasets/bio.train --predictions-file predictions/bio.train.predictions
python3 classify.py --mode test --model-file trained/bio.adaboost.model --data datasets/bio.dev --predictions-file predictions/bio.dev.predictions
python3 classify.py --mode test --model-file trained/bio.adaboost.model --data datasets/bio.test --predictions-file predictions/bio.test.predictions
python3 compute_accuracy.py datasets/bio.train predictions/bio.train.predictions
python3 compute_accuracy.py datasets/bio.dev predictions/bio.dev.predictions
python3 compute_accuracy.py datasets/bio.test predictions/bio.test.predictions