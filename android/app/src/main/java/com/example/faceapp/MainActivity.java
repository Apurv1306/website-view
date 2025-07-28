package com.example.faceapp;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import android.webkit.WebView;
import com.chaquo.python.PyObject;
import com.chaquo.python.Python;

public class MainActivity extends AppCompatActivity {
  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    Python.start(new AndroidPlatform(this));
    setContentView(R.layout.activity_main);

    // Start Flask server in Python
    PyObject flaskModule = Python.getInstance()
                                .getModule("backend.finalapkcode");
    flaskModule.callAttr("app", "run", 
      python.getBuiltins().get("dict")
        .callAttr("host","127.0.0.1","port",5000,"debug",false)
    );

    WebView webView = findViewById(R.id.webview);
    webView.getSettings().setJavaScriptEnabled(true);
    webView.loadUrl("http://127.0.0.1:5000/app.html");
  }
}
