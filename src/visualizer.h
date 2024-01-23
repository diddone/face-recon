#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

#include "bfm_manager.h"
#include "face_mesh.h"
#include "shader_class.h"
#include "utils.h"

#include <iostream>
#include <utility>

class Visualizer {
public:
    Visualizer(int _argc, char *_argv) : argc(_argc) {
        *argv = _argv;
        transformationMatrix = Eigen::Matrix4d::Identity();
        initializeOpenGL();

        imageShader = Shader("../src/shaders/4.1.texture.vs",
                             "../src/shaders/4.1.texture.fs");
        faceMeshShader = Shader("../src/shaders/4.1.texture.mesh.vs",
                                "../src/shaders/4.1.texture.mesh.fs");
    }

    Mesh face_mesh;
    Eigen::Matrix4d transformationMatrix;
    int imageWidth, imageHeight;

    // settings
    const unsigned int SCR_WIDTH = 800;
    const unsigned int SCR_HEIGHT = 800;

private:
    GLFWwindow *window;
    unsigned int VBO, VAO, EBO;
    unsigned int imageTexture;
    Shader imageShader;
    Shader faceMeshShader;
    int argc;
    char *argv[];

    bool initializeOpenGL() {
        // glfw: initialize and configure
        // ------------------------------
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

        // glfw window creation
        // --------------------
        window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Help", NULL, NULL);
        if (window == nullptr) {
            std::cout << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return false;
        }
        glfwMakeContextCurrent(window);
        //        glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

        // glad: load all OpenGL function pointers
        // ---------------------------------------
        if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
            std::cout << "Failed to initialize GLAD" << std::endl;
            return false;
        }

        return true;
    }

public:
    void setTransformationMatrix(Eigen::Matrix4d _transformationMatrix) {
        transformationMatrix = std::move(_transformationMatrix);
    }

    void setupImage(std::string image_location) {
        // load image, create imageTexture and generate mipmaps
        int nrChannels;

        stbi_set_flip_vertically_on_load(true);

        unsigned char *data =
                stbi_load(image_location.data(), &imageWidth, &imageHeight, &nrChannels, 3);

        float aspect_ratio = float(imageHeight) / float(imageWidth);

        // set up vertex data (and buffer(s)) and configure vertex attributes
        // ------------------------------------------------------------------
        // comments got f***** up because of code reformatting
        float imageFrameVertices[] = {
                // positions                 // colors           // imageTexture coords
                1.0f, aspect_ratio, 0.0f, 1.0f,
                0.0f, 0.0f, 1.0f, 1.0f, // top right
                1.0f, -aspect_ratio, 0.0f, 0.0f,
                1.0f, 0.0f, 1.0f, 0.0f, // bottom right
                -1.0f, -aspect_ratio, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f, // bottom left
                -1.0f, aspect_ratio, 0.0f, 1.0f,
                1.0f, 0.0f, 0.0f, 1.0f // top left
        };
        unsigned int imageFrameIndices[] = {
                0, 1, 3, // first triangle
                1, 2, 3  // second triangle
        };

        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(imageFrameVertices),
                     imageFrameVertices, GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(imageFrameIndices),
                     imageFrameIndices, GL_STATIC_DRAW);

        // position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                              (void *) 0);
        glEnableVertexAttribArray(0);
        // color attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                              (void *) (3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        // imageTexture coord attribute
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                              (void *) (6 * sizeof(float)));
        glEnableVertexAttribArray(2);

        // load and create a imageTexture
        // -------------------------
        glGenTextures(1, &imageTexture);
        glBindTexture(GL_TEXTURE_2D,
                      imageTexture); // all upcoming GL_TEXTURE_2D operations now
        // have effect on this imageTexture object
        // set the imageTexture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                        GL_REPEAT); // set imageTexture wrapping to GL_REPEAT
        // (default wrapping method)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        // set imageTexture filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                        GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        if (data) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imageWidth, imageHeight, 0, GL_RGB,
                         GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D);
        } else {
            std::cout << "Failed to load imageTexture" << std::endl;
        }
        stbi_image_free(data);
    }

    bool setupFace(shared_ptr<BfmManager> pBfmManager) {
        double scale = 0.0010931;
        Eigen::Matrix3d rotation;
        rotation << 0.999895, 0.0143569, -0.00170876,
                0.0144556, -0.99044, 0.137185,
                0.000277125, -0.137195, -0.990544;
        Eigen::Vector3d translation(-0.0958164, -0.0159176, 0.833721);

        transformationMatrix.block<3,3>(0,0).diagonal() *= scale;
        transformationMatrix.block<3,3>(0,0) *= rotation;
        transformationMatrix.block<3,1>(0,3) = translation;


//        float left = -250, right = 250, top = 250, bottom = -250;
//        float zFar = 1000;
//        float zNear = 0.01;
//
//        Eigen::Matrix4d projectionMatrix;
//        projectionMatrix << 2 / (right - left), 0, 0, 0, 0, 2 / (top - bottom), 0,
//                -(top + bottom) / (top - bottom), 0, 0, -2 / (zFar - zNear),
//                -(zFar + zNear) / (zFar - zNear), 0, 0, 0, 1;

        std::vector<Vertex> face_vertices;
        std::vector<unsigned int> face_indices;

        Matrix3d R = pBfmManager->getMatR() * pBfmManager->getScale();
        Vector3d t = pBfmManager->getVecT();


        for (size_t iVertex = 0; iVertex < pBfmManager->m_nVertices; iVertex++) {
            double x, y, z;
            float r, g, b;

            x = float(pBfmManager->m_vecCurrentBlendshape(iVertex * 3));
            y = float(pBfmManager->m_vecCurrentBlendshape(iVertex * 3 + 1));
            z = float(pBfmManager->m_vecCurrentBlendshape(iVertex * 3 + 2));

            r = float(pBfmManager->m_vecCurrentTex(iVertex * 3));
            g = float(pBfmManager->m_vecCurrentTex(iVertex * 3 + 1));
            b = float(pBfmManager->m_vecCurrentTex(iVertex * 3 + 2));

            face_vertices.push_back({x, y, z, r, g, b});
        }

        for (size_t iIndex = 0; iIndex < pBfmManager->m_nFaces; iIndex++) {
            face_indices.push_back(pBfmManager->m_vecTriangleList(iIndex * 3));
            face_indices.push_back(pBfmManager->m_vecTriangleList(iIndex * 3 + 1));
            face_indices.push_back(pBfmManager->m_vecTriangleList(iIndex * 3 + 2));
        }

        face_mesh = Mesh(face_vertices, face_indices);

        return true;
    }

    bool shouldRenderFrame() { return !glfwWindowShouldClose(window); }

    void setupFrame() {
        // input
        // -----
        processInput(window);

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    void renderImage() {

        // bind Texture
        glBindTexture(GL_TEXTURE_2D, imageTexture);

        // render container
        imageShader.use();

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    }

    void renderFaceMesh() {
        faceMeshShader.use();
        faceMeshShader.setMatrix("transformation", transformationMatrix.cast<float>());

        face_mesh.Draw(faceMeshShader);
    }

    void finishFrame() {

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved
        // etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    void closeOpenGL() {

        // optional: de-allocate all resources once they've outlived their purpose:
        // ------------------------------------------------------------------------
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);

        // glfw: terminate, clearing all previously allocated GLFW resources.
        // ------------------------------------------------------------------
        glfwTerminate();
    };


// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
    void processInput(GLFWwindow *window) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);
    }

//// glfw: whenever the window size changed (by OS or user resize) this callback function executes
//// ---------------------------------------------------------------------------------------------
//    void framebuffer_size_callback(GLFWwindow* window, int width, int height)
//    {
//        // make sure the viewport matches the new window dimensions; note that width and
//        // height will be significantly larger than specified on retina displays.
//        glViewport(0, 0, width, height);
//    }
};