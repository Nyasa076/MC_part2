package com.example.mc_part2

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.Button
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Surface
import androidx.compose.material.Text
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.mc_part2.ml.MobilenetQuantV1224
import com.example.mc_part2.ui.theme.MC_part2Theme
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.InputStream

class MainActivity : ComponentActivity() {
    private lateinit var bitmap: Bitmap
    var context = this
    var maxinx = mutableStateOf(0)
    var maxlabel = mutableStateOf("")

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode == Activity.RESULT_OK && requestCode == 1000) {
            data?.data?.let { uri ->
                val inputStream: InputStream? = contentResolver.openInputStream(uri)
                bitmap = BitmapFactory.decodeStream(inputStream)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        var process_img = ImageProcessor.Builder()
            .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        setContent {
            MC_part2Theme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colors.background
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Button(
                            onClick = {
                                val intent = Intent(Intent.ACTION_PICK)
                                intent.type = "image/*"
                                startActivityForResult(intent, 1000)
                            },
                            modifier = Modifier
                                .align(Alignment.CenterHorizontally)
                                .background(Color(0xFFf3c0e0), shape = RoundedCornerShape(8.dp))
                        ) {
                            Text("Select Image", color = Color.White)
                        }

                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(300.dp)
                                .background(Color.LightGray, shape = RoundedCornerShape(8.dp)),
                            contentAlignment = Alignment.Center
                        ) {
                            if (::bitmap.isInitialized) {
                                Image(
                                    bitmap = bitmap.asImageBitmap(),
                                    contentDescription = null,
                                    modifier = Modifier.fillMaxSize(),
                                    contentScale = ContentScale.Fit
                                )
                            } else {
                                Text("No Image Selected", color = Color.Black)
                            }
                        }

                        Button(
                            onClick = {
                                val tensorImage = TensorImage(DataType.UINT8)
                                tensorImage.load(bitmap)
                                val processedImage = process_img.process(tensorImage)

                                val model = MobilenetQuantV1224.newInstance(context)
                                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)
                                inputFeature0.loadBuffer(processedImage.buffer)

                                val outputs = model.process(inputFeature0)
                                val outputFeature0 = outputs.outputFeature0AsTensorBuffer
                                model.close()

                                outputFeature0.floatArray.forEachIndexed { index, fl ->
                                    if (fl > outputFeature0.floatArray[maxinx.value]) {
                                        maxinx.value = index
                                    }
                                }

                                val list_of_labels = resources.assets.open("labels.txt").bufferedReader().use { it.readText() }.split("\n")
                                maxlabel.value = list_of_labels[maxinx.value]
                            },
                            modifier = Modifier
                                .align(Alignment.CenterHorizontally)
                                .background(Color(0xFFf3c0e0), shape = RoundedCornerShape(8.dp))
                        ) {
                            Text("Predict", color = Color.White)
                        }

                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(8.dp)
                                .background(Color(0xFFf3c0e0), shape = RoundedCornerShape(8.dp)),
                        ) {
                            Text(
                                text = "Prediction: ${maxlabel.value}",
                                modifier = Modifier.padding(8.dp),
                                color = Color.White,
                                fontSize = 16.sp
                            )
                        }
                    }
                }
            }
        }
    }
}

