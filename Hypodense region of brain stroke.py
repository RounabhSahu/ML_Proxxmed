(None, 512, 512, 1)

	 Enc. 1 -> (None, 256, 256, 32)

		 Enc. 2 -> (None, 128, 128, 64)

			 Enc. 3 -> (None, 64, 64, 128)

				 Enc. 4 -> (None, 32, 32, 256)

					 Bridge Conv -> (None, 32, 32, 512)

				 Dec. 4 -> (None, 64, 64, 256)

			 Dec. 3 -> (None, 128, 128, 128)

		 Dec. 2 -> (None, 256, 256, 64)

	 Dec. 1 -> (None, 512, 512, 32)

(None, 512, 512, 1)
Epoch 1/200
WARNING:tensorflow:From C:\Users\anaconda3\Lib\site-packages\keras\src\utils\tf_utils.py:492: The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.

WARNING:tensorflow:From C:\Users\anaconda3\Lib\site-packages\keras\src\engine\base_layer_utils.py:384: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.

7/7 [==============================] - 12s 1s/step - loss: 0.2038 - dice_coef: 0.0153 - val_loss: 0.0182 - val_dice_coef: 0.0105
Epoch 2/200
C:\Users\Navya\anaconda3\Lib\site-packages\keras\src\engine\training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(
7/7 [==============================] - 7s 1s/step - loss: 0.0288 - dice_coef: 0.0147 - val_loss: 0.0556 - val_dice_coef: 0.0279
Epoch 3/200
7/7 [==============================] - 7s 1s/step - loss: 0.0378 - dice_coef: 0.0036 - val_loss: 0.0745 - val_dice_coef: 0.0184
Epoch 4/200
7/7 [==============================] - 7s 1s/step - loss: 0.0274 - dice_coef: 0.0130 - val_loss: 0.1071 - val_dice_coef: 0.0285
Epoch 5/200
7/7 [==============================] - 7s 1s/step - loss: 0.0643 - dice_coef: 0.0235 - val_loss: 0.0576 - val_dice_coef: 0.0464
Epoch 6/200
7/7 [==============================] - 7s 1s/step - loss: 0.0425 - dice_coef: 0.0393 - val_loss: 0.0700 - val_dice_coef: 0.0834
Epoch 7/200
7/7 [==============================] - 7s 1s/step - loss: 0.1210 - dice_coef: 0.1557 - val_loss: 0.0859 - val_dice_coef: 0.0759
Epoch 8/200
7/7 [==============================] - 8s 1s/step - loss: 0.0591 - dice_coef: 0.0425 - val_loss: 0.0479 - val_dice_coef: 0.0338
Epoch 9/200
7/7 [==============================] - 7s 1s/step - loss: 0.0712 - dice_coef: 0.0670 - val_loss: 0.0228 - val_dice_coef: 0.0219
Epoch 10/200
7/7 [==============================] - 7s 1s/step - loss: 0.0385 - dice_coef: 0.0146 - val_loss: 0.1236 - val_dice_coef: 0.1745
Epoch 11/200
7/7 [==============================] - 7s 1s/step - loss: 0.0253 - dice_coef: 0.0166 - val_loss: 0.0460 - val_dice_coef: 0.0492
Epoch 12/200
7/7 [==============================] - 7s 1s/step - loss: 0.0533 - dice_coef: 0.1920 - val_loss: 0.0458 - val_dice_coef: 0.0398
Epoch 13/200
7/7 [==============================] - 9s 1s/step - loss: 0.0612 - dice_coef: 0.1207 - val_loss: 0.2627 - val_dice_coef: 0.2340
Epoch 14/200
7/7 [==============================] - 9s 1s/step - loss: 0.0319 - dice_coef: 0.0342 - val_loss: 1.0527 - val_dice_coef: 0.0616
Epoch 15/200
7/7 [==============================] - 8s 1s/step - loss: 0.2958 - dice_coef: 0.0520 - val_loss: 0.0940 - val_dice_coef: 0.0900
Epoch 16/200
7/7 [==============================] - 8s 1s/step - loss: 0.0421 - dice_coef: 0.0480 - val_loss: 0.1112 - val_dice_coef: 0.0172
Epoch 17/200
7/7 [==============================] - 8s 1s/step - loss: 0.1255 - dice_coef: 0.0273 - val_loss: 0.0419 - val_dice_coef: 0.0215
Epoch 18/200
7/7 [==============================] - 7s 1s/step - loss: 0.0436 - dice_coef: 0.0253 - val_loss: 0.0782 - val_dice_coef: 0.0897
Epoch 19/200
7/7 [==============================] - 7s 1s/step - loss: 0.0600 - dice_coef: 0.1191 - val_loss: 0.0492 - val_dice_coef: 0.0412
Epoch 20/200
7/7 [==============================] - 7s 1s/step - loss: 0.0570 - dice_coef: 0.0570 - val_loss: 0.0909 - val_dice_coef: 0.0808
Epoch 21/200
7/7 [==============================] - 7s 1s/step - loss: 0.0417 - dice_coef: 0.0547 - val_loss: 0.0734 - val_dice_coef: 0.1776
Epoch 22/200
7/7 [==============================] - 7s 1s/step - loss: 0.0468 - dice_coef: 0.0313 - val_loss: 0.0788 - val_dice_coef: 0.0548
Epoch 23/200
7/7 [==============================] - 7s 1s/step - loss: 0.0045 - dice_coef: 0.0863 - val_loss: 0.0108 - val_dice_coef: 0.0087
Epoch 24/200
C:\Users\Navya\anaconda3\Lib\site-packages\keras\src\engine\training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(
7/7 [==============================] - 7s 1s/step - loss: 0.0658 - dice_coef: 0.1857 - val_loss: 0.0110 - val_dice_coef: 7.8476e-04
Epoch 25/200
7/7 [==============================] - 7s 1s/step - loss: 0.0759 - dice_coef: 0.0899 - val_loss: 0.0492 - val_dice_coef: 0.0526
Epoch 26/200
7/7 [==============================] - 7s 1s/step - loss: 0.0363 - dice_coef: 0.1803 - val_loss: 0.0275 - val_dice_coef: 0.1642
Epoch 27/200
7/7 [==============================] - 7s 1s/step - loss: 0.0049 - dice_coef: 0.0026 - val_loss: 0.1413 - val_dice_coef: 0.0210
Epoch 28/200
7/7 [==============================] - 7s 1s/step - loss: 0.0988 - dice_coef: 0.1783 - val_loss: 0.0322 - val_dice_coef: 0.0301
Epoch 29/200
7/7 [==============================] - 8s 1s/step - loss: 0.0454 - dice_coef: 0.0503 - val_loss: 0.0931 - val_dice_coef: 0.1229
Epoch 30/200
7/7 [==============================] - 7s 1s/step - loss: 0.0352 - dice_coef: 0.0264 - val_loss: 0.0709 - val_dice_coef: 0.0332
Epoch 31/200
7/7 [==============================] - 7s 1s/step - loss: 0.0604 - dice_coef: 0.2339 - val_loss: 0.0257 - val_dice_coef: 0.0068
Epoch 32/200
7/7 [==============================] - 7s 1s/step - loss: 0.0893 - dice_coef: 0.1903 - val_loss: 0.0430 - val_dice_coef: 0.0403
Epoch 33/200
7/7 [==============================] - 7s 1s/step - loss: 0.0942 - dice_coef: 0.1284 - val_loss: 0.0300 - val_dice_coef: 0.0242
Epoch 34/200
7/7 [==============================] - 7s 1s/step - loss: 0.0568 - dice_coef: 0.0670 - val_loss: 0.0577 - val_dice_coef: 0.0648
Epoch 35/200
7/7 [==============================] - 7s 1s/step - loss: 0.0567 - dice_coef: 0.0662 - val_loss: 0.0647 - val_dice_coef: 0.0708
Epoch 36/200
7/7 [==============================] - 7s 1s/step - loss: 0.0704 - dice_coef: 0.2344 - val_loss: 0.0656 - val_dice_coef: 0.1575
Epoch 37/200
7/7 [==============================] - 8s 1s/step - loss: 0.0934 - dice_coef: 0.2585 - val_loss: 0.0478 - val_dice_coef: 0.1476
Epoch 38/200
7/7 [==============================] - 7s 1s/step - loss: 0.0261 - dice_coef: 0.0757 - val_loss: 0.0262 - val_dice_coef: 0.0104
Epoch 39/200
7/7 [==============================] - 7s 1s/step - loss: 0.0476 - dice_coef: 0.0408 - val_loss: 0.1035 - val_dice_coef: 0.0297
Epoch 40/200
7/7 [==============================] - 7s 1s/step - loss: 0.0237 - dice_coef: 0.0170 - val_loss: 0.0605 - val_dice_coef: 0.2699
Epoch 41/200
7/7 [==============================] - 7s 1s/step - loss: 0.0612 - dice_coef: 0.1741 - val_loss: 0.0564 - val_dice_coef: 0.0658
Epoch 42/200
7/7 [==============================] - 7s 1s/step - loss: 0.0729 - dice_coef: 0.1007 - val_loss: 0.0429 - val_dice_coef: 0.0715
Epoch 43/200
7/7 [==============================] - 7s 1s/step - loss: 0.0275 - dice_coef: 0.0358 - val_loss: 0.0653 - val_dice_coef: 0.1676
Epoch 44/200
7/7 [==============================] - 7s 1s/step - loss: 0.0299 - dice_coef: 0.1467 - val_loss: 0.0767 - val_dice_coef: 0.0389
Epoch 45/200
7/7 [==============================] - 7s 1s/step - loss: 0.0347 - dice_coef: 0.0505 - val_loss: 0.0270 - val_dice_coef: 0.0173
Epoch 46/200
7/7 [==============================] - 7s 1s/step - loss: 0.0617 - dice_coef: 0.0541 - val_loss: 0.0633 - val_dice_coef: 0.0572
Epoch 47/200
7/7 [==============================] - 7s 1s/step - loss: 0.0275 - dice_coef: 0.0265 - val_loss: 0.0264 - val_dice_coef: 0.0899
Epoch 48/200
7/7 [==============================] - 7s 1s/step - loss: 0.0377 - dice_coef: 0.0190 - val_loss: 0.0270 - val_dice_coef: 0.0210
Epoch 49/200
7/7 [==============================] - 7s 1s/step - loss: 0.0343 - dice_coef: 0.1599 - val_loss: 0.0206 - val_dice_coef: 0.0662
Epoch 50/200
7/7 [==============================] - 7s 1s/step - loss: 0.0375 - dice_coef: 0.1622 - val_loss: 0.0498 - val_dice_coef: 0.0510
Epoch 51/200
7/7 [==============================] - 7s 1s/step - loss: 0.0440 - dice_coef: 0.1738 - val_loss: 0.0641 - val_dice_coef: 0.1145
Epoch 52/200
7/7 [==============================] - 7s 1s/step - loss: 0.0390 - dice_coef: 0.1830 - val_loss: 0.0895 - val_dice_coef: 0.0592
Epoch 53/200
7/7 [==============================] - 7s 1s/step - loss: 0.0743 - dice_coef: 0.1769 - val_loss: 0.0467 - val_dice_coef: 0.0638
Epoch 54/200
7/7 [==============================] - 7s 1s/step - loss: 0.0569 - dice_coef: 0.0543 - val_loss: 0.0913 - val_dice_coef: 0.0988
Epoch 55/200
7/7 [==============================] - 7s 1s/step - loss: 0.0485 - dice_coef: 0.0806 - val_loss: 0.0397 - val_dice_coef: 0.1445
Epoch 56/200
7/7 [==============================] - 7s 1s/step - loss: 0.0789 - dice_coef: 0.0979 - val_loss: 0.0645 - val_dice_coef: 0.0980
Epoch 57/200
7/7 [==============================] - 7s 1s/step - loss: 0.0950 - dice_coef: 0.0965 - val_loss: 0.0149 - val_dice_coef: 0.0222
Epoch 58/200
7/7 [==============================] - 7s 1s/step - loss: 0.0259 - dice_coef: 0.0238 - val_loss: 0.0317 - val_dice_coef: 0.1706
Epoch 59/200
7/7 [==============================] - 7s 1s/step - loss: 0.0260 - dice_coef: 0.0112 - val_loss: 0.0452 - val_dice_coef: 0.1688
Epoch 60/200
7/7 [==============================] - 7s 1s/step - loss: 0.0417 - dice_coef: 0.1608 - val_loss: 0.0928 - val_dice_coef: 0.0563
Epoch 61/200
7/7 [==============================] - 7s 1s/step - loss: 0.0477 - dice_coef: 0.1806 - val_loss: 0.0896 - val_dice_coef: 0.1174
Epoch 62/200
7/7 [==============================] - 7s 1s/step - loss: 0.0928 - dice_coef: 0.1279 - val_loss: 0.0380 - val_dice_coef: 0.1655
Epoch 63/200
7/7 [==============================] - 7s 1s/step - loss: 0.0669 - dice_coef: 0.0402 - val_loss: 0.0350 - val_dice_coef: 0.0863
Epoch 64/200
7/7 [==============================] - 7s 1s/step - loss: 0.0412 - dice_coef: 0.0335 - val_loss: 0.0687 - val_dice_coef: 0.0796
Epoch 65/200
7/7 [==============================] - 7s 1s/step - loss: 0.0735 - dice_coef: 0.0831 - val_loss: 0.0322 - val_dice_coef: 0.0305
Epoch 66/200
7/7 [==============================] - 7s 1s/step - loss: 0.0383 - dice_coef: 0.0475 - val_loss: 0.0171 - val_dice_coef: 0.1551
Epoch 67/200
7/7 [==============================] - 7s 1s/step - loss: 0.0463 - dice_coef: 0.1011 - val_loss: 0.0577 - val_dice_coef: 0.0686
Epoch 68/200
7/7 [==============================] - 7s 1s/step - loss: 0.0124 - dice_coef: 0.1583 - val_loss: 0.0585 - val_dice_coef: 0.0641
Epoch 69/200
7/7 [==============================] - 7s 1s/step - loss: 0.0449 - dice_coef: 0.1781 - val_loss: 0.0201 - val_dice_coef: 0.0298
Epoch 70/200
7/7 [==============================] - 7s 1s/step - loss: 0.0443 - dice_coef: 0.0321 - val_loss: 0.0633 - val_dice_coef: 0.0731
Epoch 71/200
7/7 [==============================] - 7s 1s/step - loss: 0.0626 - dice_coef: 0.0746 - val_loss: 0.0331 - val_dice_coef: 0.0248
Epoch 72/200
7/7 [==============================] - 7s 1s/step - loss: 0.0569 - dice_coef: 0.0466 - val_loss: 0.0207 - val_dice_coef: 0.0255
Epoch 73/200
7/7 [==============================] - 7s 1s/step - loss: 0.0183 - dice_coef: 0.0826 - val_loss: 0.0643 - val_dice_coef: 0.0255
Epoch 74/200
7/7 [==============================] - 7s 1s/step - loss: 0.0112 - dice_coef: 0.1641 - val_loss: 0.0988 - val_dice_coef: 0.0523
Epoch 75/200
7/7 [==============================] - 7s 1s/step - loss: 0.0662 - dice_coef: 0.0463 - val_loss: 0.0681 - val_dice_coef: 0.1075
Epoch 76/200
7/7 [==============================] - 7s 1s/step - loss: 0.0459 - dice_coef: 0.0914 - val_loss: 0.0379 - val_dice_coef: 0.1670
Epoch 77/200
7/7 [==============================] - 7s 1s/step - loss: 0.0380 - dice_coef: 0.4580 - val_loss: 0.0441 - val_dice_coef: 0.3169
Epoch 78/200
7/7 [==============================] - 7s 1s/step - loss: 0.0664 - dice_coef: 0.1707 - val_loss: 0.0885 - val_dice_coef: 0.0948
Epoch 79/200
7/7 [==============================] - 7s 1s/step - loss: 0.0053 - dice_coef: 0.2712 - val_loss: 0.0775 - val_dice_coef: 0.0694
Epoch 80/200
7/7 [==============================] - 7s 1s/step - loss: 0.0602 - dice_coef: 0.2089 - val_loss: 0.0332 - val_dice_coef: 0.1168
Epoch 81/200
7/7 [==============================] - 7s 1s/step - loss: 0.0620 - dice_coef: 0.0703 - val_loss: 0.0223 - val_dice_coef: 0.0154
Epoch 82/200
7/7 [==============================] - 7s 1s/step - loss: 0.0821 - dice_coef: 0.0914 - val_loss: 0.0557 - val_dice_coef: 0.0632
Epoch 83/200
7/7 [==============================] - 7s 1s/step - loss: 0.0511 - dice_coef: 0.0577 - val_loss: 0.0589 - val_dice_coef: 0.0750
Epoch 84/200
7/7 [==============================] - 7s 1s/step - loss: 0.0827 - dice_coef: 0.0202 - val_loss: 0.0572 - val_dice_coef: 0.1354
Epoch 85/200
7/7 [==============================] - 7s 1s/step - loss: 0.0632 - dice_coef: 0.0574 - val_loss: 0.0877 - val_dice_coef: 0.0231
Epoch 86/200
7/7 [==============================] - 7s 1s/step - loss: 0.1642 - dice_coef: 0.0350 - val_loss: 0.0467 - val_dice_coef: 0.0454
Epoch 87/200
7/7 [==============================] - 7s 1s/step - loss: 0.0168 - dice_coef: 0.0173 - val_loss: 0.0440 - val_dice_coef: 0.0271
Epoch 88/200
7/7 [==============================] - 7s 1s/step - loss: 0.0285 - dice_coef: 0.0088 - val_loss: 0.0813 - val_dice_coef: 0.0505
Epoch 89/200
7/7 [==============================] - 7s 1s/step - loss: 0.0017 - dice_coef: 0.2611 - val_loss: 0.0162 - val_dice_coef: 0.0272
Epoch 90/200
7/7 [==============================] - 7s 1s/step - loss: 0.0192 - dice_coef: 0.3039 - val_loss: 0.0451 - val_dice_coef: 0.0204
Epoch 91/200
7/7 [==============================] - 8s 1s/step - loss: 0.0598 - dice_coef: 0.0465 - val_loss: 0.0432 - val_dice_coef: 0.1271
Epoch 92/200
7/7 [==============================] - 8s 1s/step - loss: 0.0508 - dice_coef: 0.1769 - val_loss: 0.0269 - val_dice_coef: 0.0505
Epoch 93/200
7/7 [==============================] - 7s 1s/step - loss: 0.0808 - dice_coef: 0.1043 - val_loss: 0.0539 - val_dice_coef: 0.0678
Epoch 94/200
7/7 [==============================] - 7s 1s/step - loss: 0.0151 - dice_coef: 8.2269e-04 - val_loss: 0.0416 - val_dice_coef: 0.0321
Epoch 95/200
7/7 [==============================] - 7s 1s/step - loss: 0.0316 - dice_coef: 0.2900 - val_loss: 0.0665 - val_dice_coef: 0.0129
Epoch 96/200
7/7 [==============================] - 7s 1s/step - loss: 0.0923 - dice_coef: 0.0753 - val_loss: 0.0186 - val_dice_coef: 0.0124
Epoch 97/200
7/7 [==============================] - 7s 1s/step - loss: 0.0497 - dice_coef: 0.0507 - val_loss: 0.0398 - val_dice_coef: 0.0320
Epoch 98/200
7/7 [==============================] - 7s 1s/step - loss: 0.0507 - dice_coef: 0.0648 - val_loss: 0.0319 - val_dice_coef: 0.1676
Epoch 99/200
7/7 [==============================] - 7s 1s/step - loss: 0.0798 - dice_coef: 0.0472 - val_loss: 0.0485 - val_dice_coef: 0.3152
Epoch 100/200
7/7 [==============================] - 7s 1s/step - loss: 0.0476 - dice_coef: 0.1837 - val_loss: 0.0882 - val_dice_coef: 0.0797
Epoch 101/200
7/7 [==============================] - 7s 1s/step - loss: 0.0829 - dice_coef: 0.1036 - val_loss: 0.0137 - val_dice_coef: 0.0229
Epoch 102/200
7/7 [==============================] - 7s 1s/step - loss: 0.0995 - dice_coef: 0.0532 - val_loss: 0.0327 - val_dice_coef: 0.1692
Epoch 103/200
7/7 [==============================] - 8s 1s/step - loss: 0.0515 - dice_coef: 0.0554 - val_loss: 0.0081 - val_dice_coef: 0.0025
Epoch 104/200
C:\Users\Navya\anaconda3\Lib\site-packages\keras\src\engine\training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(
7/7 [==============================] - 8s 1s/step - loss: 0.0188 - dice_coef: 0.0171 - val_loss: 0.0646 - val_dice_coef: 0.0372
Epoch 105/200
7/7 [==============================] - 8s 1s/step - loss: 0.0678 - dice_coef: 0.3138 - val_loss: 0.0293 - val_dice_coef: 0.1257
Epoch 106/200
7/7 [==============================] - 7s 1s/step - loss: 0.0694 - dice_coef: 0.1480 - val_loss: 0.0586 - val_dice_coef: 0.0965
Epoch 107/200
7/7 [==============================] - 8s 1s/step - loss: 0.0086 - dice_coef: 0.1465 - val_loss: 0.0791 - val_dice_coef: 0.0403
Epoch 108/200
7/7 [==============================] - 7s 1s/step - loss: 0.0303 - dice_coef: 0.0425 - val_loss: 0.0727 - val_dice_coef: 0.0333
Epoch 109/200
7/7 [==============================] - 7s 1s/step - loss: 0.0596 - dice_coef: 0.0341 - val_loss: 0.0688 - val_dice_coef: 0.2179
Epoch 110/200
7/7 [==============================] - 7s 1s/step - loss: 0.0626 - dice_coef: 0.2578 - val_loss: 0.0163 - val_dice_coef: 0.0219
Epoch 111/200
7/7 [==============================] - 7s 1s/step - loss: 0.0654 - dice_coef: 0.0644 - val_loss: 0.0369 - val_dice_coef: 0.0657
Epoch 112/200
7/7 [==============================] - 7s 1s/step - loss: 0.0563 - dice_coef: 0.2095 - val_loss: 0.0606 - val_dice_coef: 0.0865
Epoch 113/200
7/7 [==============================] - 7s 1s/step - loss: 0.0277 - dice_coef: 0.1857 - val_loss: 0.0232 - val_dice_coef: 0.2624
Epoch 114/200
7/7 [==============================] - 7s 1s/step - loss: 0.0638 - dice_coef: 0.1130 - val_loss: 0.0283 - val_dice_coef: 0.3193
Epoch 115/200
7/7 [==============================] - 7s 1s/step - loss: 0.0510 - dice_coef: 0.0465 - val_loss: 0.0725 - val_dice_coef: 0.1650
Epoch 116/200
7/7 [==============================] - 7s 1s/step - loss: 0.0265 - dice_coef: 0.1657 - val_loss: 0.0576 - val_dice_coef: 0.0363
Epoch 117/200
7/7 [==============================] - 7s 1s/step - loss: 0.0600 - dice_coef: 0.1752 - val_loss: 0.0320 - val_dice_coef: 0.1918
Epoch 118/200
7/7 [==============================] - 7s 1s/step - loss: 0.0450 - dice_coef: 0.0624 - val_loss: 0.0321 - val_dice_coef: 0.1854
Epoch 119/200
7/7 [==============================] - 7s 1s/step - loss: 0.0361 - dice_coef: 0.0346 - val_loss: 0.0486 - val_dice_coef: 0.1897
Epoch 120/200
7/7 [==============================] - 7s 1s/step - loss: 0.0337 - dice_coef: 0.0316 - val_loss: 0.0379 - val_dice_coef: 0.0361
Epoch 121/200
7/7 [==============================] - 7s 1s/step - loss: 0.0134 - dice_coef: 0.0075 - val_loss: 0.0509 - val_dice_coef: 0.2977
Epoch 122/200
7/7 [==============================] - 7s 1s/step - loss: 0.0254 - dice_coef: 0.2966 - val_loss: 0.0939 - val_dice_coef: 0.1754
Epoch 123/200
7/7 [==============================] - 7s 1s/step - loss: 0.0216 - dice_coef: 0.0165 - val_loss: 0.0528 - val_dice_coef: 0.0582
Epoch 124/200
7/7 [==============================] - 7s 1s/step - loss: 0.0305 - dice_coef: 0.0364 - val_loss: 0.0541 - val_dice_coef: 0.3431
Epoch 125/200
7/7 [==============================] - 7s 1s/step - loss: 0.0296 - dice_coef: 0.2773 - val_loss: 0.0386 - val_dice_coef: 0.0660
Epoch 126/200
7/7 [==============================] - 7s 1s/step - loss: 0.0869 - dice_coef: 0.1590 - val_loss: 0.0259 - val_dice_coef: 0.1831
Epoch 127/200
7/7 [==============================] - 7s 1s/step - loss: 0.0568 - dice_coef: 0.0979 - val_loss: 0.0639 - val_dice_coef: 0.2336
Epoch 128/200
7/7 [==============================] - 7s 1s/step - loss: 0.0546 - dice_coef: 0.0856 - val_loss: 0.0346 - val_dice_coef: 0.2138
Epoch 129/200
7/7 [==============================] - 7s 1s/step - loss: 0.0301 - dice_coef: 0.0556 - val_loss: 0.0645 - val_dice_coef: 0.1115
Epoch 130/200
7/7 [==============================] - 7s 1s/step - loss: 0.0405 - dice_coef: 0.2985 - val_loss: 0.0525 - val_dice_coef: 0.1775
Epoch 131/200
7/7 [==============================] - 7s 1s/step - loss: 0.0495 - dice_coef: 0.2029 - val_loss: 0.0527 - val_dice_coef: 0.1143
Epoch 132/200
7/7 [==============================] - 7s 1s/step - loss: 0.0468 - dice_coef: 0.0498 - val_loss: 0.0409 - val_dice_coef: 0.1699
Epoch 133/200
7/7 [==============================] - 7s 1s/step - loss: 0.0391 - dice_coef: 0.1690 - val_loss: 0.0409 - val_dice_coef: 0.2196
Epoch 134/200
7/7 [==============================] - 7s 1s/step - loss: 0.0193 - dice_coef: 0.1183 - val_loss: 0.0537 - val_dice_coef: 0.0547
Epoch 135/200
7/7 [==============================] - 7s 1s/step - loss: 0.0389 - dice_coef: 0.2918 - val_loss: 0.0684 - val_dice_coef: 0.3450
Epoch 136/200
7/7 [==============================] - 7s 1s/step - loss: 0.0363 - dice_coef: 0.0882 - val_loss: 0.0492 - val_dice_coef: 0.2431
Epoch 137/200
7/7 [==============================] - 7s 1s/step - loss: 0.1578 - dice_coef: 0.1546 - val_loss: 0.0721 - val_dice_coef: 0.0936
Epoch 138/200
7/7 [==============================] - 7s 1s/step - loss: 0.0450 - dice_coef: 0.0634 - val_loss: 0.0314 - val_dice_coef: 0.0385
Epoch 139/200
7/7 [==============================] - 8s 1s/step - loss: 0.0697 - dice_coef: 0.0800 - val_loss: 0.0306 - val_dice_coef: 0.3168
Epoch 140/200
7/7 [==============================] - 7s 1s/step - loss: 0.0750 - dice_coef: 0.1676 - val_loss: 0.0601 - val_dice_coef: 0.2202
Epoch 141/200
7/7 [==============================] - 7s 1s/step - loss: 0.0579 - dice_coef: 0.1591 - val_loss: 0.0178 - val_dice_coef: 0.1404
Epoch 142/200
7/7 [==============================] - 7s 1s/step - loss: 0.0470 - dice_coef: 0.3258 - val_loss: 0.0398 - val_dice_coef: 0.0496
Epoch 143/200
7/7 [==============================] - 7s 1s/step - loss: 0.0396 - dice_coef: 0.2216 - val_loss: 0.0209 - val_dice_coef: 0.2062
Epoch 144/200
7/7 [==============================] - 7s 1s/step - loss: 0.0396 - dice_coef: 0.0334 - val_loss: 0.1241 - val_dice_coef: 0.1007
Epoch 145/200
7/7 [==============================] - 7s 1s/step - loss: 0.0027 - dice_coef: 0.2498 - val_loss: 0.0639 - val_dice_coef: 0.1171
Epoch 146/200
7/7 [==============================] - 7s 1s/step - loss: 0.0137 - dice_coef: 0.3028 - val_loss: 0.0542 - val_dice_coef: 0.0370
Epoch 147/200
7/7 [==============================] - 7s 1s/step - loss: 0.0433 - dice_coef: 0.2244 - val_loss: 0.0453 - val_dice_coef: 0.3614
Epoch 148/200
7/7 [==============================] - 7s 1s/step - loss: 0.0660 - dice_coef: 0.0774 - val_loss: 0.0373 - val_dice_coef: 0.1801
Epoch 149/200
7/7 [==============================] - 7s 1s/step - loss: 0.0077 - dice_coef: 0.1531 - val_loss: 0.0394 - val_dice_coef: 0.0679
Epoch 150/200
7/7 [==============================] - 7s 1s/step - loss: 0.0584 - dice_coef: 0.2067 - val_loss: 0.0530 - val_dice_coef: 0.0910
Epoch 151/200
7/7 [==============================] - 7s 1s/step - loss: 0.0210 - dice_coef: 0.1670 - val_loss: 0.0654 - val_dice_coef: 0.0659
Epoch 152/200
7/7 [==============================] - 7s 1s/step - loss: 0.0522 - dice_coef: 0.0757 - val_loss: 0.0508 - val_dice_coef: 0.2324
Epoch 153/200
7/7 [==============================] - 7s 1s/step - loss: 0.0638 - dice_coef: 0.0968 - val_loss: 0.0302 - val_dice_coef: 0.0948
Epoch 154/200
7/7 [==============================] - 7s 1s/step - loss: 0.0369 - dice_coef: 0.3572 - val_loss: 0.0175 - val_dice_coef: 0.3256
Epoch 155/200
7/7 [==============================] - 7s 1s/step - loss: 0.0378 - dice_coef: 0.0713 - val_loss: 0.0066 - val_dice_coef: 0.0066
Epoch 156/200
C:\Users\Navya\anaconda3\Lib\site-packages\keras\src\engine\training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(
7/7 [==============================] - 7s 1s/step - loss: 0.0299 - dice_coef: 0.0139 - val_loss: 0.0321 - val_dice_coef: 0.0597
Epoch 157/200
7/7 [==============================] - 7s 1s/step - loss: 0.0274 - dice_coef: 0.0142 - val_loss: 0.0618 - val_dice_coef: 0.2084
Epoch 158/200
7/7 [==============================] - 7s 1s/step - loss: 0.0558 - dice_coef: 0.2281 - val_loss: 0.0565 - val_dice_coef: 0.2618
Epoch 159/200
7/7 [==============================] - 7s 1s/step - loss: 0.0248 - dice_coef: 0.0634 - val_loss: 0.0682 - val_dice_coef: 0.0491
Epoch 160/200
7/7 [==============================] - 7s 1s/step - loss: 0.0209 - dice_coef: 0.0603 - val_loss: 0.0273 - val_dice_coef: 0.1621
Epoch 161/200
7/7 [==============================] - 7s 1s/step - loss: 0.0182 - dice_coef: 0.3085 - val_loss: 0.0056 - val_dice_coef: 0.0530
Epoch 162/200
C:\Users\Navya\anaconda3\Lib\site-packages\keras\src\engine\training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(
7/7 [==============================] - 7s 1s/step - loss: 0.0542 - dice_coef: 0.1058 - val_loss: 0.0510 - val_dice_coef: 0.0922
Epoch 163/200
7/7 [==============================] - 8s 1s/step - loss: 0.0616 - dice_coef: 0.0692 - val_loss: 0.0184 - val_dice_coef: 0.0490
Epoch 164/200
7/7 [==============================] - 8s 1s/step - loss: 0.0036 - dice_coef: 0.0118 - val_loss: 0.0401 - val_dice_coef: 0.1100
Epoch 165/200
7/7 [==============================] - 7s 1s/step - loss: 0.0717 - dice_coef: 0.3736 - val_loss: 0.0688 - val_dice_coef: 0.2824
Epoch 166/200
7/7 [==============================] - 7s 1s/step - loss: 0.0616 - dice_coef: 0.0950 - val_loss: 0.0757 - val_dice_coef: 0.1025
Epoch 167/200
7/7 [==============================] - 7s 1s/step - loss: 0.0754 - dice_coef: 0.1197 - val_loss: 0.0459 - val_dice_coef: 0.1041
Epoch 168/200
7/7 [==============================] - 7s 1s/step - loss: 0.0588 - dice_coef: 0.0705 - val_loss: 0.0398 - val_dice_coef: 0.2172
Epoch 169/200
7/7 [==============================] - 7s 1s/step - loss: 0.0644 - dice_coef: 0.3806 - val_loss: 0.0455 - val_dice_coef: 0.2259
Epoch 170/200
7/7 [==============================] - 7s 1s/step - loss: 0.0124 - dice_coef: 9.9336e-04 - val_loss: 0.0293 - val_dice_coef: 0.1683
Epoch 171/200
7/7 [==============================] - 7s 1s/step - loss: 0.0503 - dice_coef: 0.0576 - val_loss: 0.0621 - val_dice_coef: 0.0611
Epoch 172/200
7/7 [==============================] - 7s 1s/step - loss: 0.0520 - dice_coef: 0.0965 - val_loss: 0.0667 - val_dice_coef: 0.4077
Epoch 173/200
7/7 [==============================] - 8s 1s/step - loss: 0.0339 - dice_coef: 0.1882 - val_loss: 0.0058 - val_dice_coef: 0.1445
Epoch 174/200
7/7 [==============================] - 7s 1s/step - loss: 0.0777 - dice_coef: 0.0734 - val_loss: 0.0181 - val_dice_coef: 0.0370
Epoch 175/200
7/7 [==============================] - 8s 1s/step - loss: 0.0325 - dice_coef: 0.2066 - val_loss: 0.0301 - val_dice_coef: 0.0574
Epoch 176/200
7/7 [==============================] - 8s 1s/step - loss: 0.0714 - dice_coef: 0.2448 - val_loss: 0.0016 - val_dice_coef: 0.0039
Epoch 177/200
C:\Users\anaconda3\Lib\site-packages\keras\src\engine\training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(
7/7 [==============================] - 7s 1s/step - loss: 0.0405 - dice_coef: 0.0593 - val_loss: 0.0497 - val_dice_coef: 0.0956
Epoch 178/200
7/7 [==============================] - 7s 1s/step - loss: 0.0268 - dice_coef: 0.1797 - val_loss: 0.0546 - val_dice_coef: 0.0770
Epoch 179/200
7/7 [==============================] - 7s 1s/step - loss: 0.0218 - dice_coef: 0.2058 - val_loss: 0.0420 - val_dice_coef: 0.0383
Epoch 180/200
7/7 [==============================] - 7s 1s/step - loss: 0.0302 - dice_coef: 0.3337 - val_loss: 0.0227 - val_dice_coef: 0.2464
Epoch 181/200
7/7 [==============================] - 7s 1s/step - loss: 0.0221 - dice_coef: 0.0757 - val_loss: 0.0163 - val_dice_coef: 0.3187
Epoch 182/200
7/7 [==============================] - 7s 1s/step - loss: 0.1022 - dice_coef: 0.1040 - val_loss: 0.0234 - val_dice_coef: 0.2316
Epoch 183/200
7/7 [==============================] - 8s 1s/step - loss: 0.0298 - dice_coef: 0.0996 - val_loss: 0.0563 - val_dice_coef: 0.2589
Epoch 184/200
7/7 [==============================] - 7s 1s/step - loss: 0.0350 - dice_coef: 0.1913 - val_loss: 0.0251 - val_dice_coef: 0.2167
Epoch 185/200
7/7 [==============================] - 7s 1s/step - loss: 0.0123 - dice_coef: 0.1651 - val_loss: 0.0479 - val_dice_coef: 0.1939
Epoch 186/200
7/7 [==============================] - 7s 1s/step - loss: 0.0983 - dice_coef: 0.4651 - val_loss: 0.0539 - val_dice_coef: 0.2985
Epoch 187/200
7/7 [==============================] - 7s 1s/step - loss: 0.0704 - dice_coef: 0.2671 - val_loss: 0.0259 - val_dice_coef: 0.1993
Epoch 188/200
7/7 [==============================] - 7s 1s/step - loss: 0.0631 - dice_coef: 0.0450 - val_loss: 0.0897 - val_dice_coef: 0.2065
Epoch 189/200
7/7 [==============================] - 7s 1s/step - loss: 0.0422 - dice_coef: 0.2012 - val_loss: 0.0516 - val_dice_coef: 0.2105
Epoch 190/200
7/7 [==============================] - 7s 1s/step - loss: 0.0292 - dice_coef: 0.1578 - val_loss: 0.0489 - val_dice_coef: 0.0832
Epoch 191/200
7/7 [==============================] - 7s 1s/step - loss: 0.0457 - dice_coef: 0.2651 - val_loss: 0.0245 - val_dice_coef: 0.1607
Epoch 192/200
7/7 [==============================] - 7s 1s/step - loss: 0.0593 - dice_coef: 0.4808 - val_loss: 0.0472 - val_dice_coef: 0.0764
Epoch 193/200
7/7 [==============================] - 7s 1s/step - loss: 0.0295 - dice_coef: 0.0775 - val_loss: 0.0376 - val_dice_coef: 0.0871
Epoch 194/200
7/7 [==============================] - 7s 1s/step - loss: 0.0443 - dice_coef: 0.2340 - val_loss: 0.0433 - val_dice_coef: 0.0516
Epoch 195/200
7/7 [==============================] - 7s 1s/step - loss: 0.0887 - dice_coef: 0.1432 - val_loss: 0.0380 - val_dice_coef: 0.0735
Epoch 196/200
7/7 [==============================] - 7s 1s/step - loss: 0.0582 - dice_coef: 0.1679 - val_loss: 0.0169 - val_dice_coef: 0.0813
Epoch 197/200
7/7 [==============================] - 7s 1s/step - loss: 0.0413 - dice_coef: 0.0468 - val_loss: 0.0629 - val_dice_coef: 0.0622
Epoch 198/200
7/7 [==============================] - 7s 1s/step - loss: 0.0612 - dice_coef: 0.1135 - val_loss: 0.0285 - val_dice_coef: 0.0188
Epoch 199/200
7/7 [==============================] - 7s 1s/step - loss: 0.0205 - dice_coef: 0.0200 - val_loss: 0.0569 - val_dice_coef: 0.1228
Epoch 200/200
7/7 [==============================] - 7s 1s/step - loss: 0.0468 - dice_coef: 0.1901 - val_loss: 0.0727 - val_dice_coef: 0.0587

