import tensorflow as tf
from tensorflow.keras import layers, models
import time
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import shutil


# 数据加载和预处理
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# 1. LeNet-5模型
def create_lenet():
    model = models.Sequential([
        layers.Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# 2. 简化版AlexNet（因为MNIST图像太小，需要调整）
def create_alexnet():
    model = models.Sequential([
        # 输入层
        layers.Input(shape=(28, 28, 1)),  # MNIST 图像是 28x28x1
        
        # 第一个卷积块 - 减小卷积核和步长
        layers.Conv2D(96, 3, strides=1, padding='same', activation='relu'),
        layers.MaxPooling2D(2, strides=2),  # 输出 14x14
        layers.BatchNormalization(),
        
        # 第二个卷积块
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2, strides=2),  # 输出 7x7
        layers.BatchNormalization(),
        
        # 第三个卷积块
        layers.Conv2D(384, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        
        # 第四个卷积块
        layers.Conv2D(384, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        
        # 第五个卷积块
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2, strides=2),  # 输出 3x3
        layers.BatchNormalization(),
        
        # 全连接层
        layers.Flatten(),
        layers.Dense(2048, activation='relu'),  # 减小全连接层的大小
        layers.Dropout(0.5),
        layers.Dense(1024, activation='relu'),  # 减小全连接层的大小
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # 10个类别（MNIST数据集）
    ])
    
    return model


# 3. 传统BP神经网络
def create_bp_network():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# 数据增强
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1)
])

# 评估函数
def evaluate_model(model_fn, model_name, use_augmentation=False):
    model = model_fn()
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    # 记录开始时间
    start_time = time.time()
    
    # 训练
    if use_augmentation:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_dataset = train_dataset.shuffle(60000).batch(32)
        train_dataset = train_dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y))
        history = model.fit(train_dataset, epochs=10, 
                          validation_data=(test_images, test_labels))
    else:
        history = model.fit(train_images, train_labels, epochs=10,
                          validation_data=(test_images, test_labels))
    
    # 记录训练时间
    training_time = time.time() - start_time
    
    # 测试性能
    start_time = time.time()
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    inference_time = time.time() - start_time
    
    return {
        'model_name': model_name,
        'accuracy': test_acc,
        'training_time': training_time,
        'inference_time': inference_time,
        'history': history.history
    }

def evaluate_model_with_tensorboard(model_fn, model_name, use_augmentation=False):
    # 确保目录存在
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", "fit", current_time + "_" + model_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Created log directory: {log_dir}")  # 调试信息

    # 创建训练和验证数据的日志记录器
    train_log_dir = os.path.join(log_dir, 'train')
    val_log_dir = os.path.join(log_dir, 'validation')
    os.makedirs(train_log_dir, exist_ok=True)
    os.makedirs(val_log_dir, exist_ok=True)

    # 创建 TensorBoard 回调
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch'
    )

    # 创建自定义回调来记录更多指标
    class CustomCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super(CustomCallback, self).__init__()
            self.train_writer = tf.summary.create_file_writer(train_log_dir)
            self.val_writer = tf.summary.create_file_writer(val_log_dir)

        def on_epoch_end(self, epoch, logs=None):
            with self.train_writer.as_default():
                tf.summary.scalar('loss', logs['loss'], step=epoch)
                tf.summary.scalar('accuracy', logs['accuracy'], step=epoch)

            with self.val_writer.as_default():
                tf.summary.scalar('loss', logs['val_loss'], step=epoch)
                tf.summary.scalar('accuracy', logs['val_accuracy'], step=epoch)

    # 创建和编译模型
    model = model_fn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 打印模型结构
    model.summary()

    # 准备回调列表
    callbacks = [
        CustomCallback(),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,  # 每个epoch记录直方图
            profile_batch=0  # 禁用性能分析以避免CUPTI错误
        )
    ]

    # 训练模型
    if use_augmentation:
        # 创建数据增强层
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])

        # 创建增强的数据集
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_dataset = train_dataset.shuffle(60000).batch(32)
        train_dataset = train_dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y)
        )

        history = model.fit(
            train_dataset,
            epochs=10,
            validation_data=(test_images, test_labels),
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            train_images,
            train_labels,
            batch_size=32,
            epochs=10,
            validation_data=(test_images, test_labels),
            callbacks=callbacks,
            verbose=1
        )

    # 评估模型
    test_results = model.evaluate(test_images, test_labels, verbose=1)
    
    # 记录最终测试结果
    with tf.summary.create_file_writer(os.path.join(log_dir, 'test')).as_default():
        for metric, value in zip(model.metrics_names, test_results):
            tf.summary.scalar(f'final_{metric}', value, step=0)

    print(f"\nTensorBoard logs saved to: {log_dir}")
    print("To view the training progress, run:")
    print(f"tensorboard --logdir={log_dir}")

    return {
        'model_name': model_name,
        'accuracy': test_results[1],  # accuracy is typically the second metric
        'history': history.history,
        'tensorboard_log_dir': log_dir
    }



def plot_training_history(results):
    plt.figure(figsize=(15, 5))
    
    # 准确率对比
    plt.subplot(1, 2, 1)
    for result in results:
        plt.plot(result['history']['accuracy'], label=f"{result['model_name']} (Training)")
        plt.plot(result['history']['val_accuracy'], label=f"{result['model_name']} (Validation)")
    plt.title('Model Accuracy over Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # 损失值对比
    plt.subplot(1, 2, 2)
    for result in results:
        plt.plot(result['history']['loss'], label=f"{result['model_name']} (Training)")
        plt.plot(result['history']['val_loss'], label=f"{result['model_name']} (Validation)")
    plt.title('Model Loss over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 性能对比表格
def print_performance_comparison(results):
    print("\nPerformance Comparison:")
    print("=" * 80)
    print(f"{'Model Name':<30} {'Accuracy':<10} {'Training Time':<15} {'Inference Time':<15}")
    print("-" * 80)
    for result in results:
        print(f"{result['model_name']:<30} "
              f"{result['accuracy']*100:>8.2f}% "
              f"{result['training_time']:>13.2f}s "
              f"{result['inference_time']:>13.2f}s")

def test_custom_images(model, image_paths):
    """
    测试自定义的手写数字图片
    """
    results = []
    for path in image_paths:
        # 加载图片并预处理
        img = tf.keras.preprocessing.image.load_img(path, color_mode='grayscale', target_size=(28, 28))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255

        # 预测
        start_time = time.time()
        predictions = model.predict(img_array)
        inference_time = time.time() - start_time
        
        predicted_digit = np.argmax(predictions[0])
        confidence = predictions[0][predicted_digit]
        
        results.append({
            'path': path,
            'predicted': predicted_digit,
            'confidence': confidence,
            'inference_time': inference_time
        })
    
    return results

# 生成实验报告数据
def generate_report_data(results):
    report = {
        'model_comparison': {
            'accuracy': {},
            'training_time': {},
            'inference_time': {},
        },
        'augmentation_impact': {},
        'best_model': None
    }
    
    for result in results:
        model_name = result['model_name']
        report['model_comparison']['accuracy'][model_name] = result['accuracy']
        report['model_comparison']['training_time'][model_name] = result['training_time']
        report['model_comparison']['inference_time'][model_name] = result['inference_time']
    
    return report


# 运行实验
if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs('logs/fit', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 定义要评估的模型
    models_to_evaluate = [
        (create_lenet, 'LeNet-5'),
        (create_alexnet, 'AlexNet'),
        (create_bp_network, 'BP Network')
    ]

    try:
        # 1. 运行基础评估
        print("\n=== Running Basic Evaluation ===")
        results = []
        for model_fn, name in models_to_evaluate:
            print(f"\nEvaluating {name}...")
            # 不使用数据增强
            result = evaluate_model(model_fn, f"{name}")
            results.append(result)
            
            # 使用数据增强
            print(f"Evaluating {name} with data augmentation...")
            result = evaluate_model(model_fn, f"{name} (with augmentation)", True)
            results.append(result)

        # 2. 绘制训练历史
        plot_training_history(results)

        # 3. 打印性能对比
        print_performance_comparison(results)

        # 4. 找出并打印最佳模型
        best_model = None
        best_accuracy = 0
        for result in results:
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_model_name = result['model_name']

        print(f"\nBest performing model: {best_model_name} with accuracy: {best_accuracy*100:.2f}%")

        # 5. 运行带TensorBoard的评估（可选）
        use_tensorboard = input("\nDo you want to run evaluation with TensorBoard? (y/n): ").lower() == 'y'
        if use_tensorboard:
            print("\n=== Running Evaluation with TensorBoard ===")
            base_log_dir = os.path.join("logs", "fit")
            os.makedirs(base_log_dir, exist_ok=True)
            
            # 清空之前的日志
            for item in os.listdir(base_log_dir):
                item_path = os.path.join(base_log_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            
            tensorboard_results = []
            for model_fn, name in models_to_evaluate:
                print(f"\nEvaluating {name} with TensorBoard...")
                result = evaluate_model_with_tensorboard(model_fn, name)
                tensorboard_results.append(result)
                
                print(f"\nEvaluating {name} with TensorBoard and data augmentation...")
                result = evaluate_model_with_tensorboard(model_fn, f"{name}_augmented", True)
                tensorboard_results.append(result)

            print("\nAll TensorBoard evaluations completed.")
            print(f"\nTo view training progress in TensorBoard, run:")
            print(f"tensorboard --logdir={base_log_dir}")
            
            # 等待用户确认
            input("\nPress Enter to continue...")

        # 6. 生成实验报告数据
        report_data = generate_report_data(results)
        print("\nExperiment report data generated successfully.")

    except Exception as e:
        print(f"\nAn error occurred during evaluation: {str(e)}")
        raise

